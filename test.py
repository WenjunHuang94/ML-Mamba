import requests
import torch

from PIL import Image
from pathlib import Path

from mlmamba import load

hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# In case your GPU does not support bf16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # torch.bfloat16

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
model_id = "mlmamba+3b"
vlm = load(model_id, hf_token=hf_token)

# # Obtain the state_dect of the entire model
# state_dict = vlm.state_dict()

## Extract parameters from llm_backbone and projector sections and adjust keys
## This is because the module parameters corresponding to load_date-dict_ff ("state-spaces/mamba2-2.7b") are MLMambaVLM.llm_backbone. llm, so we also need to store them here
# filtered_state_dict = {
#     "llm_backbone": {k[len("llm_backbone.llm."):]: v for k, v in state_dict.items() if k.startswith("llm_backbone.llm.")},
#     "mlp": {k[len("mlp."):]: v for k, v in state_dict.items() if k.startswith("mlp.")},
#     "bidirectional_mamba": {k[len("bidirectional_mamba."):]: v for k, v in state_dict.items() if k.startswith("bidirectional_mamba.")},
#     "projector": {k[len("projector."):]: v for k, v in state_dict.items() if k.startswith("projector.")}
# }
# # Save extracted parameters
# save_path = "./vlm_projector_mamba2_2.7b_v5_model.pth"
# torch.save({"model": filtered_state_dict}, save_path)
#
# model_state_dict = torch.load(save_path, map_location="cpu")["model"]
#
# # View modules and parameters
# module_keys = set()
# for key in model_state_dict.keys():
#     module_name = key.split('.')[0]
#     module_keys.add(module_name)
#
# # Print module name
# print("Modules in the state dict:")
# for module in sorted(module_keys):
#     print(module)
#
# # Check the parameter data types of the entire model
# print("Data types of all parameters in MLMambaVLM:")
# for name, param in vlm.named_parameters():
#     print(f"{name}: {param.dtype}")


vlm.to(device, dtype=dtype)  # 不转dtype=bfoat16输出会出错



# Download an image and specify a prompt
#image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
#image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# image = Image.open('test7.png').convert("RGB")
# user_prompt = "What are the things I should be cautious about when I visit here?"
# image = Image.open('test1.png').convert("RGB")
# user_prompt = "Is the bicycle parked at the right of the dog?"

# Build prompt
# prompt_builder = vlm.get_prompt_builder()
# prompt_builder.add_turn(role="human", message=user_prompt)
# prompt_text = prompt_builder.get_prompt()  # 经过prompt_builder添加过特殊start、end字符
#prompt_text = user_prompt

# Generate!
# generated_text = vlm.generate(
#     image,
#     prompt_text,
#     use_cache=True,
#     do_sample=True,
#     #temperature=0.4,
#     temperature=1.0,
#     #max_new_tokens=512,
#     max_new_tokens=512, # 这个会影响生成的长度
# )


# 图片文件名列表
image_files = [f'test0.png', f'test1.png', f'test2.png', f'test3.png', f'test4.png', f'test5.png', f'test6.png']

# 用户提示列表
user_prompts = [
    "Provide a detailed description of this image",
    "Is the bicycle parked on the right side of the dog?",
    "What's unusual about this photo?",
    "What should I pay attention to when I come here?",
    "Can I swim here?",
    "What's going on in this picture?",
    "What's going on in this picture?"
]

# 生成参数
generate_params = {
    'use_cache': True,
    'do_sample': True,
    'temperature': 1.0,
    'max_new_tokens': 256
}



for image_file, user_prompt in zip(image_files, user_prompts):
    # 打开并转换图像
    image = Image.open("pic/" + image_file).convert("RGB")

    # 创建prompt
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=user_prompt)
    prompt_text = prompt_builder.get_prompt()

    # 更新生成参数
    generate_params['image'] = image
    generate_params['prompt_text'] = prompt_text

    # 生成文本
    generated_text = vlm.generate(**generate_params)

    print(f'Image {image_file}: generated_text = ', generated_text)