"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


'''
数据集预处理的，特别是在多模态（语言和图像）的训练场景中:
（1）数据格式化：将输入的instances（每个实例包含input_ids（文本）、labels（目标标签）和pixel_values（图像数据））转换为模型所需的格式。
它首先对文本部分进行pad_sequence处理，确保所有样本的长度不超过model_max_length，并填充pad_token_id作为填充值
（2）处理多模态：根据实例中pixel_values的存在与否，将它们整理成一个multimodal_indices列表，用于区分哪些样本是多模态的，哪些是语言模型的纯文本数据。
对于纯文本样本，它会用dummy_pixel_values填充图像部分
（3）填充像素值：如果pixel_values是torch.Tensor类型，它会将所有样本的图像数据堆叠成一个大的张量；如果pixel_values是字典类型，
它会为每个键（如rgb、depth等）创建一个堆叠的张量。对于缺失的图像数据，使用dummy_pixel_values填充
'''
@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)  # self.pad_token_id=0,来批次数据长度对齐
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)  # IGNORE_INDEX=-100

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)  # 目前我们计算损失时, 实际没用到attention_mask

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(  # 有图片的就是多模态
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )
