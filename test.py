import requests
import torch

from PIL import Image
from pathlib import Path

from cobra import load

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# In case your GPU does not support bf16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # torch.bfloat16

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "cobra+3b"
vlm = load(model_id, hf_token=hf_token)

# # 获取整个模型的state_dict
# state_dict = vlm.state_dict()
#
# # 提取llm_backbone和projector部分的参数并调整键
# # 这里是因为load_state_dict_hf("state-spaces/mamba2-2.7b")时对应的是CobraVLM.llm_backbone.llm的模块参数，所以我们也要对应的只存这里
# filtered_state_dict = {
#     "llm_backbone": {k[len("llm_backbone.llm."):]: v for k, v in state_dict.items() if k.startswith("llm_backbone.llm.")},
#     "projector": {k[len("projector."):]: v for k, v in state_dict.items() if k.startswith("projector.")}
# }
#
# # 保存提取的参数
# save_path = "./vlm_mamba2_130mv2_model.pth"
# torch.save({"model": filtered_state_dict}, save_path)
#
# model_state_dict = torch.load(save_path, map_location="cpu")["model"]
#
# # 查看模块和参数
# module_keys = set()
# for key in model_state_dict.keys():
#     # 提取模块名称（键的前缀部分）
#     module_name = key.split('.')[0]
#     module_keys.add(module_name)
#
# # 打印模块名称
# print("Modules in the state dict:")
# for module in sorted(module_keys):
#     print(module)

# # 检查整个模型的参数数据类型
# print("Data types of all parameters in CobraVLM:")
# for name, param in vlm.named_parameters():
#     print(f"{name}: {param.dtype}")


vlm.to(device, dtype=dtype) # 不转dtype=bfoat16输出会出错
#vlm.to(device)



# Download an image and specify a prompt
#image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
#image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

image = Image.open('beignets-task-guide.png').convert("RGB")
user_prompt = "What is going on in this image?"
#user_prompt = "what is Sun Yat sen University?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()  # 经过prompt_builder添加过特殊start、end字符
#prompt_text = user_prompt

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    use_cache=True,
    do_sample=True,
    #temperature=0.4,
    temperature=1.0,
    #max_new_tokens=512,
    max_new_tokens=512, # 这个会影响生成的长度
)

print('generated_text = ', generated_text)
