import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA(nn.Module):
    def __init__(self, original_dim, lora_rank, target_dim):
        super(LoRA, self).__init__()
        self.lora_A = nn.Parameter(torch.randn(original_dim, lora_rank))
        self.lora_B = nn.Parameter(torch.randn(lora_rank, target_dim))

    def forward(self, x):
        return x @ (self.lora_A @ self.lora_B)

class SharedLoRA(nn.Module):
    def __init__(self, dim, lora_rank):
        super(SharedLoRA, self).__init__()
        self.lora = LoRA(dim, lora_rank, dim)

    def forward(self, x):
        return self.lora(x)

class SpecificLoRAAttention(nn.Module):
    def __init__(self, dim, lora_rank, shared_lora):
        super(SpecificLoRAAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.shared_lora = shared_lora
        self.specific_lora_query = LoRA(dim, lora_rank, dim)
        self.specific_lora_key = LoRA(dim, lora_rank, dim)
        self.specific_lora_value = LoRA(dim, lora_rank, dim)

    def forward(self, x):
        shared_output = self.shared_lora(x)
        q = self.query(x) + shared_output + self.specific_lora_query(x)
        k = self.key(x) + shared_output + self.specific_lora_key(x)
        v = self.value(x) + shared_output + self.specific_lora_value(x)
        scores = q @ k.transpose(-2, -1) / (x.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return attn @ v

# 示例输入
batch_size = 2
seq_len = 10
dim = 32
lora_rank = 4

# 模拟输入数据
x = torch.randn(batch_size, seq_len, dim)

# 实例化共享LORA和特定注意力LORA
shared_lora = SharedLoRA(dim, lora_rank)

# 创建若干特定的注意力LORA
attention1 = SpecificLoRAAttention(dim, lora_rank, shared_lora)

# 前向传播
output1 = attention1(x)

print("Output1 Shape:", output1.shape)
