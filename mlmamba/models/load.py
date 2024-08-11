"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from mlmamba.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from mlmamba.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from mlmamba.models.vlms import MLMambaVLM
from mlmamba.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "huangwenjun1994/ML-Mamba"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `mlmamba.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None
):
    """Loads a pretrained MLMambaVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        assert (config_json := run_dir / "config.json").exists(), f"Missing `config.json` for `{run_dir = }`"
        assert (checkpoint_pt := run_dir / "checkpoints" / "latest-checkpoint.pt").exists(), "Missing checkpoint!"
    else:  # 进这
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `mlmamba.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")  # model_id_or_path='mlmamba+3b'

        #config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"config.json", cache_dir=cache_dir)
        config_json = os.path.join(os.path.dirname(__file__), '..', 'conf', 'config.json')  # 里面需要改llm_backbone_id

        # checkpoint_pt = hf_hub_download(
        #     repo_id=HF_HUB_REPO,
        #     filename=f"latest-checkpoint.pt",
        #     cache_dir=cache_dir,
        #     use_auth_token=hf_token
        # )


        # 注意是aligin还是finetune!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        checkpoint_pt = '/home/disk2/cobra+3b+stage-finetune+x7/step-332649-epoch-00-loss=0.7296.pt'
        #checkpoint_pt = '/home/hwj/program/mlmamba/scripts/runs/mlmamba+3b+stage-finetune+x7/checkpoints/latest-vlm-mamba2-2.7b-checkpoint-0624.pt'


    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    model_cfg['llm_backbone_id'] = "mamba2-2.7b"  # 改为你需要的mamba2模型




    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),  # 2048
        hf_token=hf_token,
        inference_mode=True,
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],  # dinosiglip-vit-so-384px
        model_cfg["image_resize_strategy"],  # resize-naive
    )


    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint; Freezing Weights 🥶")
    vlm = MLMambaVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        llm_backbone_id=model_cfg['llm_backbone_id'],
        arch_specifier=model_cfg["arch_specifier"],
    )

    return vlm
