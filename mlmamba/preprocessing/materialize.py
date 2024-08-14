"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from mlmamba.conf import DatasetConfig
from mlmamba.models.backbones.llm.prompting import PromptBuilder
from mlmamba.models.backbones.vision import ImageTransform
from mlmamba.preprocessing.datasets import AlignDataset, FinetuneDataset
from mlmamba.util.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:


    dataset_cls = DATASET_INITIALIZER[stage]  # <class 'mlmamba.preprocessing.datasets.datasets.FinetuneDataset'>
    dataset_root_dir = dataset_cfg.dataset_root_dir  # PosixPath('data')
    collator = PaddedCollatorForLanguageModeling(  # Fill in data
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components  # annotation_json = 'download/llava-laion-cc-sbu-558k/chat.json', image_dir = 'download/llava-laion-cc-sbu-558k'
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components  # annotation_json = PosixPath('download/llava-v1.5-instruct/llava_v1_5_mix665k.json')
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,  # /data/download/llava-v1.5-instruct/llava_v1_5_mix665k.json
            dataset_root_dir / image_dir,  # /data/download/llava-v1.5-instruct
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
