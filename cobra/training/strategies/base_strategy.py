"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.vlms import CobraVLM
from cobra.overwatch import initialize_overwatch
from cobra.training.metrics import Metrics
from cobra.util import check_bloat16_supported
from cobra.util.batching_utils import SplitModalitySampler
from cobra.util.data_utils import PaddedCollatorForLanguageModeling

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

import time


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: CobraVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality": # 进这个
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(   # 用于将数据集中的样本按照模态（例如，图像和文本）和长度进行分组，以便每个设备（如GPU）在训练时看到的样本具有相似的序列长度
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        # collate_fn参数用于指定数据加载器在每个批次加载数据后如何对数据进行整合。具体来说，collate_fn参数接受一个函数作为输入，
        # 该函数将被用于对每个批次的数据进行自定义的整合操作，以便将多个样本整合成一个批次的数据。
        # 通常，collate_fn函数会接收一个包含单个样本的列表，并将其整合成一个包含整个批次数据的张量或其他数据结构。
        # 这对于处理不同大小或类型的样本数据非常有用，可以根据需要进行自定义的数据整合操作，例如填充、对齐或其他预处理步骤。
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,  # sampler: 可选参数，用于指定数据加载的采样方法，例如随机采样、有放回采样等
            collate_fn=collator,  # 可选参数，用于指定数据加载器在加载每个批次的数据后如何对数据进行整合
            # num_workers=2,  # 通过设置 num_workers 来让数据加载过程在多个子进程中并行进行，从而加快数据加载的速度
            num_workers=1,  # 方便调试用，否则debug时，数据很容易看乱，因为不同进程处理的数据不一样
            worker_init_fn=self.worker_init_fn,  # 可选参数，用于指定子进程的初始化函数
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        # 调试用！！！！！！！
        self.epochs = 1
        start_time = time.time()
        hour_counter = 0

        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()  # 这行代码将模型设置为训练模式。这通常会启用诸如Dropout等训练时特定的行为，但不会影响参数的requires_grad属性
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,  # 关闭时值应为False
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:  # self.max_steps值为空，要改掉
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())  # PosixPath('runs/cobra+3b+stage-finetune+x7')
                            dist.barrier()  # 在分布式环境中同步不同进程之间的操作的函数。当一个进程调用 dist.barrier() 时，它会被阻塞

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)


                    current_time = time.time()
                    if current_time - start_time >= 3600:  # 检查是否已经过了一个小时
                    #if True:
                        hour_counter += 1
                        self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())

                        dist.barrier()
                        start_time = time.time()  # 重置起始时间

                        #break  # 调试用

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            #if self.max_steps is None:  # 循环结束都保存
            #    self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
            #    dist.barrier()

            #torch.save(self.vlm.state_dict(), '/home/hwj/program/cobra/vlm_model.pth')
            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
            dist.barrier()
            print(22222222222222222222222222222)
