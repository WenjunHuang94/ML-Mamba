<div align="center">

 <h2><img src="./assets/logo-2.png" style='width: 3%'> ML-Mamba: Efficient Multi-Modal Large Language Model Utilizing Mamba-2</h2>


<img src="./assets/arch.png" style='width: 75%'>


</div>

## Introduction
Multimodal Large Language Models (MLLMs) have attracted much attention for their multifunctionality. However, traditional Transformer architectures incur significant overhead due to their secondary computational complexity. To address this issue, we introduce ML-Mamba, a multimodal language model, which utilizes the latest and efficient Mamba-2 model for inference. Mamba-2 is known for its linear scalability and fast processing of long sequences. We replace the Transformer-based backbone with a pre-trained Mamba-2 model and explore methods for integrating 2D visual selective scanning mechanisms into multimodal learning while also trying various visual encoders and Mamba-2 model variants. Our extensive experiments in various multimodal benchmark tests demonstrate the competitive performance of ML-Mamba and highlight the potential of state space models in multimodal tasks. The experimental results show that: (1) We empirically explore the application of 2D visual selective scanning in multimodal learning and propose the Mamba-2 Scan Connector (MSC) to enhance representational capabilities. (2)  ML-Mamba achieves performance comparable to state-of-the-art methods such as TinyLaVA and MobileVLM v2 through its linear sequential modeling while faster inference speed; (3) Compared to multimodal models utilizing Mamba-1, the Mamba-2-based ML-Mamba exhibits superior inference performance and effectiveness.


[**Installation**](#installation) | [**Usage**](#usage) | [**Training VLMs**](#training-vlms) | [**License**](#license)
---

## Installation

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require PyTorch 2.1 or greater installation instructions [can be found here](https://pytorch.org/get-started/locally/). This repository was developed and has been thoroughly tested with PyTorch 2.1.0 and Torchvision 0.16.0.

Once PyTorch has been properly installed, you can install this package locally via an editable installation

```bash
conda create -n ML-Mamba python=3.10
conda activate ML-Mamba
cd ML-Mamba
pip install -e .

# install mamba and other packages
cd causal-conv1d-main && pip install -e . && cd ..
cd mamba && pip install -e . &&  cd .. 
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# option
pip install --upgrade Pillow
pip install --upgrade numpy
pip install --upgrade huggingface_hub
pip install numpy==1.21.2
pip install --upgrade click
```

If you run into any problems during the installation process, please file a GitHub Issue.

## Usage

Once installed, loading and running inference with pretrained `ML-Mamba` models is easy:

*First, you need to create a **.hf_token** file in the ML-Mamba project directory and fill in your Huggingface token.*

```python
import requests
import torch

from PIL import Image
from pathlib import Path

from mlmamba import load

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# In case your GPU does not support bf16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "mlmamba+3b"
vlm = load(model_id, hf_token=hf_token)

vlm.to(device, dtype=dtype)
image = Image.open("pic/test0.png").convert("RGB")
user_prompt = "Provide a detailed description of this image"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    use_cache=True,
    do_sample=True,
    temperature=1.0,
    max_new_tokens=512,
)

print(f'user_prompt : {user_prompt} \ngenerated_text : {generated_text}')
```

For a complete terminal-based CLI for interacting with our VLMs, check out [scripts/generate.py]. 

---



## Training VLMs

#### Pretraining Datasets
The [LLaVa v1.5 Instruct Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) can be downloaded by the automated download script in [`scripts/preprocess.py`]:

```bash
# Download the `llava-v1.5-instruct` (Instruct Tuning) Image and Language Data (includes extra post-processing)
python scripts/preprocess.py --dataset_id "llava-v1.5-instruct" --root_dir <PATH-TO-DATA-ROOT>

# (In case you also wish to download the explicit vision-language alignment data)
python scripts/preprocess.py --dataset_id "llava-laion-cc-sbu-558k" --root_dir <PATH-TO-DATA-ROOT>
```

[LVIS-Instruct-4V](https://arxiv.org/abs/2311.07574) and [LRV-Instruct](https://arxiv.org/abs/2306.14565) can also be downloaded by the scripts in [`scripts/additional-datasets`].

#### Model Configuration & Training Script
Here's how you would train ML-Mamba follow the training recipe in our paper across 8 GPUs on a single-node: 

*First, you need to create a **.hf_token** file in the ML-Mamba/scripts project directory and fill in your Huggingface token.*

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "mamba2-2.7b" \
  --model.type "mlmamba+3b" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 8 \
  --dataset.type "llava-v15"
```

---

## ML-Mamba Evaluation

The evaluation code for the ML-Mamba project is located in the `vlm-evaluation` directory. Please follow the detailed instructions provided in `vlm-evaluation/README.md` to run the evaluation.

---

### License
This project is released under the [MIT License](LICENSE.txt)


## Acknowledgement

This repository is built based on [LLaVA](https://github.com/haotian-liu/LLaVA),  [Mamba](https://github.com/state-spaces/mamba), [Transformers](https://github.com/JLTastet/transformers/tree/mamba), and [Cobra](https://github.com/h-zhao1997/cobra) for their public code release.