import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, original_dim, lora_rank, target_dim):
        super(LoRA, self).__init__()
        self.lora_A = nn.Parameter(torch.randn(original_dim, lora_rank))
        self.lora_B = nn.Parameter(torch.randn(lora_rank, target_dim))

    def forward(self, x):
        return x @ (self.lora_A @ self.lora_B)

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
        attn = torch.softmax(scores, dim=-1)
        return attn @ v

# Instantiate the model
dim = 1024
lora_rank = 16

specific_lora_attention = SpecificLoRAAttention(dim, lora_rank)

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate parameters
query_params = count_parameters(specific_lora_attention.query)
key_params = count_parameters(specific_lora_attention.key)
value_params = count_parameters(specific_lora_attention.value)
specific_lora_query_params = count_parameters(specific_lora_attention.specific_lora_query)
specific_lora_key_params = count_parameters(specific_lora_attention.specific_lora_key)
specific_lora_value_params = count_parameters(specific_lora_attention.specific_lora_value)

print(f"Query Parameters: {query_params} ({query_params / 1e6:.2f}M)")
print(f"Key Parameters: {key_params} ({key_params / 1e6:.2f}M)")
print(f"Value Parameters: {value_params} ({value_params / 1e6:.2f}M)")
print(f"Specific LoRA Query Parameters: {specific_lora_query_params} ({specific_lora_query_params / 1e6:.2f}M)")
print(f"Specific LoRA Key Parameters: {specific_lora_key_params} ({specific_lora_key_params / 1e6:.2f}M)")
print(f"Specific LoRA Value Parameters: {specific_lora_value_params} ({specific_lora_value_params / 1e6:.2f}M)")
