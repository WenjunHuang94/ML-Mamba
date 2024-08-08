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

# # 获取整个模型的state_dict
# state_dict = vlm.state_dict()
#
# # 提取llm_backbone和projector部分的参数并调整键
# # 这里是因为load_state_dict_hf("state-spaces/mamba2-2.7b")时对应的是MLMambaVLM.llm_backbone.llm的模块参数，所以我们也要对应的只存这里
# filtered_state_dict = {
#     "llm_backbone": {k[len("llm_backbone.llm."):]: v for k, v in state_dict.items() if k.startswith("llm_backbone.llm.")},
#     "mlp": {k[len("mlp."):]: v for k, v in state_dict.items() if k.startswith("mlp.")},
#     "bidirectional_mamba": {k[len("bidirectional_mamba."):]: v for k, v in state_dict.items() if k.startswith("bidirectional_mamba.")},
#     "projector": {k[len("projector."):]: v for k, v in state_dict.items() if k.startswith("projector.")}
# }
# # 保存提取的参数
# save_path = "./vlm_projector_mamba2_2.7b_v5_model.pth"
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
#
# # 检查整个模型的参数数据类型
# print("Data types of all parameters in MLMambaVLM:")
# for name, param in vlm.named_parameters():
#     print(f"{name}: {param.dtype}")


vlm.to(device, dtype=dtype) # 不转dtype=bfoat16输出会出错



# Download an image and specify a prompt
#image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
#image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# image = Image.open('test7.png').convert("RGB")
# user_prompt = "What are the things I should be cautious about when I visit here?"
# image = Image.open('test1.png').convert("RGB")
# user_prompt = "Is the bicycle parked at the right of the dog?"
#user_prompt = "what is Sun Yat sen University?"

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


# import time
#
# # 设置生成参数
# generate_params = {
#     'image': image,
#     'prompt_text': prompt_text,
#     'use_cache': True,
#     'do_sample': True,
#     'temperature': 1.0,
#     'max_new_tokens': 256
# }
#
# # 记录总时间
# total_time = 0
# num_iterations = 50
#
# # 分词器（假设你有一个分词器来计算令牌数）
# def count_tokens(text):
#     # 示例分词器函数，实际使用时需要替换为你的分词器
#     return len(text.split())
#
#
#
# # 开始计时
# start_time = time.time()
#
# total_tokens = 0
#
# for i in range(num_iterations):
#     # 生成文本
#     generated_text = vlm.generate(**generate_params)
#
#     # 计算生成文本中的令牌数量
#     #tokens = count_tokens(generated_text)
#     #total_tokens += 256
#
#     # print(f'Iteration {i + 1}: generated_text = ', generated_text)
#     # print(f'Iteration {i + 1}: len(generated_text) = ', len(generated_text))
#     #print(f'Iteration {i + 1}: tokens = ', tokens)
#
# # 结束计时
# end_time = time.time()
# total_time = (end_time - start_time)
#
# # 计算平均每次生成的时间
# average_time = total_time / num_iterations
#
# # 每秒生成的令牌数量
# Eval_avg = 256 / average_time
#
# print(f'Average time per generation: {average_time:.4f} seconds')
# print(f'Eval_avg (tokens per second): {Eval_avg:.2f}')
#
# print(f'generated_text = ', generated_text)


#print('generated_text = ', generated_text)



