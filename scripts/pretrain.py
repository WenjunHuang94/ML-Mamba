"""
pretrain.py

Pretraining script for MLMamba VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).


Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from mlmamba.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from mlmamba.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from mlmamba.overwatch import initialize_overwatch
from mlmamba.preprocessing import get_dataset_and_collator
from mlmamba.training import Metrics, get_train_strategy
from mlmamba.util import set_global_seed

import argparse

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`mlmamba/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.MLMAMBA_3B.model_id)
    )

    # DatasetConfig (`mlmamba/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)  # 换成LVIS-Instruct-4V
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load (for `finetune`)
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "mlmamba"                                # Name of W&B project (default: `mlmamba`)
    wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        if self.stage == "align":  # 对齐
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):  # 微调
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--vision_backbone_id", type=str, default="dinosiglip-vit-so-384px")
    # parser.add_argument("--image_resize_strategy", type=str, default="resize-naive")
    # parser.add_argument("--llm_backbone_id", type=str, default="mamba-2.8b-zephyr")
    # parser.add_argument("--model_type", type=str, default="mlmamba+3b")
    # parser.add_argument("--finetune_global_batch_size", type=int, default=128)
    # parser.add_argument("--finetune_per_device_batch_size", type=int, default=8)
    # parser.add_argument("--dataset_type", type=str, default="llava-lvis4v-lrv")
    #
    # args = parser.parse_args()
    #
    # cfg.model.vision_backbone_id = args.vision_backbone_id
    # cfg.model.image_resize_strategy = args.image_resize_strategy
    # cfg.model.llm_backbone_id = args.llm_backbone_id
    # cfg.model.type = args.model_type
    # cfg.model.finetune_global_batch_size = args.finetune_global_batch_size
    # cfg.model.finetune_per_device_batch_size = args.finetune_per_device_batch_size

    # cfg.dataset.type = args.dataset_type

    cfg.global_batch_size = 2  # Fill in the data batch during training
    cfg.per_device_batch_size = 2  # Fill in the data batch during training

    # （1）注意要去配置里修改下llm_backbone_id!!!
    # （2）注意save_checkpoint里修改下保存的文件名!!!
    cfg.stage = "finetune"  # finetune or align

    # Manually fill in the checkpoint path for ML-Mamba!!!
    cfg.pretrained_checkpoint = '/home/hwj/program/ML-Mamba/scripts/runs/mlmamba+3b+stage-finetune+x7/checkpoints/latest-checkpoint.pt'

    #cfg.max_steps = 100

    import torch
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29504'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'  # If it is a single GPU training
    os.environ['LOCAL_RANK'] = '0'
    dist.init_process_group(backend='nccl')

    overwatch.info("MLMamba VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    print('cfg = ', cfg)
    print('cfg.hf_token = ', cfg.hf_token)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    print('hf_token = ', hf_token)
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)  # 使用相同的随机数种子，确保可复现性
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)  # run_dir = PosixPath('runs/mlmamba+3b+stage-finetune+x7')
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone --> on CPU, in Full Precision (initializing Tokenizer + handling special tokens if necessary)
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating MLMambaVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        cfg.model.llm_backbone_id,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint (depends on stage, config)
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)  # 加载了Vision模块、mamba模块预训练参数，MLP没有加载预训练

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.stage,
        cfg.dataset,  # LLaVa_V15_Config(dataset_id='llava-v15', align_stage_components=(PosixPath('download/llava-laion-cc-sbu-558k/chat.json'), PosixPath('download/llava-laion-cc-sbu-558k')), finetune_stage_components=(PosixPath('download/llava-v1.5-instruct/llava_v1_5_mix665k.json'), PosixPath('download/llava-v1.5-instruct')), dataset_root_dir=PosixPath('data'))
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,  # <class 'mlmamba.models.backbones.llm.prompting.zephyr_prompter.ZephyrChatPromptBuilder'>
        default_image_resolution=vision_backbone.default_image_resolution,  # (3, 384, 384)
        padding_side=tokenizer.padding_side,  # 'right'
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,  # 'fsdp-full-shard'
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,  # 2
        max_steps=cfg.max_steps,  # self.max_steps值为空，要改掉！！！
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))  # 设置训练过程中的一些参数和策略

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    #train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)
    try:
        # 尝试运行训练代码
        train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)
    except torch.cuda.OutOfMemoryError as e:
        print("CUDA insufficient video memory, reducing batch size and trying again...")
        cfg.global_batch_size //= 2
        cfg.per_device_batch_size //= 2

        # 重新初始化 train_strategy
        train_strategy = get_train_strategy(
            train_strategy=cfg.train_strategy,
            vlm=vlm,
            device_id=device_id,
            epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            global_batch_size=cfg.global_batch_size,
            per_device_batch_size=cfg.per_device_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
            enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
            reduce_in_full_precision=cfg.model.reduce_in_full_precision,
            worker_init_fn=worker_init_fn,
        )

        # 重新运行 setup
        train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

        # 重新初始化 metrics
        metrics = Metrics(
            cfg.trackers,
            cfg.run_id,
            run_dir,
            draccus.encode(cfg),
            cfg.stage,
            wandb_project=cfg.wandb_project,
            wandb_entity=cfg.wandb_entity,
            grad_accumulation_steps=train_strategy.grad_accumulation_steps,
        )

        # 重新运行训练代码
        train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)


    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()
