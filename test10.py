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
    def __init__(self, dim, lora_rank):
        super(SpecificLoRAAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.specific_lora_query = LoRA(dim, lora_rank, dim)
        self.specific_lora_key = LoRA(dim, lora_rank, dim)
        self.specific_lora_value = LoRA(dim, lora_rank, dim)

    def forward(self, x, shared_output):
        q = self.query(x) + shared_output + self.specific_lora_query(x)
        k = self.key(x) + shared_output + self.specific_lora_key(x)
        v = self.value(x) + shared_output + self.specific_lora_value(x)
        scores = q @ k.transpose(-2, -1) / (x.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return attn @ v

class ParallelLoRAAttention(nn.Module):
    def __init__(self, dim, lora_rank, num_specific):
        super(ParallelLoRAAttention, self).__init__()
        self.shared_lora = SharedLoRA(dim, lora_rank)
        self.specific_attentions = nn.ModuleList([
            SpecificLoRAAttention(dim, lora_rank) for _ in range(num_specific)
        ])

    def forward(self, x):
        shared_output = self.shared_lora(x)
        outputs = [attention(x, shared_output) for attention in self.specific_attentions]
        # Combine outputs (e.g., sum them up or concatenate)
        combined_output = sum(outputs)  # or torch.cat(outputs, dim=-1) for concatenation
        return combined_output

# 示例输入
batch_size = 2
seq_len = 10
dim = 32
lora_rank = 4
num_specific = 3

# 模拟输入数据
x = torch.randn(batch_size, seq_len, dim)

# 实例化并行LORA注意力模块
parallel_attention = ParallelLoRAAttention(dim, lora_rank, num_specific)

# 前向传播
output = parallel_attention(x)

print("Parallel Output Shape:", output.shape)
