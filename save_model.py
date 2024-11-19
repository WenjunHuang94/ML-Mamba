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

#
# state_dict = vlm.state_dict()
#
#

# filtered_state_dict = {
#     "llm_backbone": {k[len("llm_backbone.llm."):]: v for k, v in state_dict.items() if k.startswith("llm_backbone.llm.")},
#     "mlp": {k[len("mlp."):]: v for k, v in state_dict.items() if k.startswith("mlp.")},
#     "bidirectional_mamba": {k[len("bidirectional_mamba."):]: v for k, v in state_dict.items() if k.startswith("bidirectional_mamba.")},
#     "projector": {k[len("projector."):]: v for k, v in state_dict.items() if k.startswith("projector.")},
#     "cross_attentions": {k[len("cross_attentions."):]: v for k, v in state_dict.items() if k.startswith("cross_attentions.")},
#     "parallel_attention": {k[len("parallel_attention."):]: v for k, v in state_dict.items() if k.startswith("parallel_attention.")}
# }

# save_path = "./ML-Mamba-1029.pth"
# torch.save({"model": filtered_state_dict}, save_path)
#
# model_state_dict = torch.load(save_path, map_location="cpu")["model"]
#

# module_keys = set()
# for key in model_state_dict.keys():

#     module_name = key.split('.')[0]
#     module_keys.add(module_name)
#

# print("Modules in the state dict:")
# for module in sorted(module_keys):
#     print(module)
#

# print("Data types of all parameters in MLMambaVLM:")
# for name, param in vlm.named_parameters():
#     print(f"{name}: {param.dtype}")

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