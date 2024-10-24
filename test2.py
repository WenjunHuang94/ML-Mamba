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
vlm.to(device, dtype=dtype) # 不转dtype=bfoat16输出会出错



import time

image = Image.open("pic/" + "test0.png").convert("RGB")
# 创建prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message="Provide a detailed description of this image")
prompt_text = prompt_builder.get_prompt()

# 设置生成参数
generate_params = {
    'image': image,
    'prompt_text': prompt_text,
    'use_cache': True,
    'do_sample': True,
    'temperature': 1.0,
    'max_new_tokens': 256
}

generated_text = vlm.generate(**generate_params)
print('generated_text = ', generated_text)


# 记录总时间
total_time = 0
num_iterations = 2

# 开始计时
start_time = time.time()


for i in range(num_iterations):
    # 生成文本
    generated_text = vlm.generate(**generate_params)


# 结束计时
end_time = time.time()
total_time = (end_time - start_time)

# 计算平均每次生成的时间
average_time = total_time / num_iterations

# 每秒生成的令牌数量
Eval_avg = 256 / average_time

print(f'Average time per generation: {average_time:.4f} seconds')
print(f'Eval_avg (tokens per second): {Eval_avg:.2f}')

print(f'generated_text = ', generated_text)




