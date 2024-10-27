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

class LoRAAttention(nn.Module):
    def __init__(self, dim, lora_rank):
        super(LoRAAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.lora_query = LoRA(dim, lora_rank, dim)
        self.lora_key = LoRA(dim, lora_rank, dim)
        self.lora_value = LoRA(dim, lora_rank, dim)

    def forward(self, x):
        q = self.query(x) + self.lora_query(x)
        k = self.key(x) + self.lora_key(x)
        v = self.value(x) + self.lora_value(x)
        scores = q @ k.transpose(-2, -1) / (x.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return attn @ v

class LoRAFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, lora_rank):
        super(LoRAFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.lora_ff = LoRA(dim, lora_rank, hidden_dim)

    def forward(self, x):
        out = self.linear1(x) + self.lora_ff(x)
        out = F.relu(out)
        return self.linear2(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, lora_rank):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = LoRAAttention(dim, lora_rank)
        self.feed_forward = LoRAFeedForward(dim, hidden_dim, lora_rank)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        return self.norm2(x + ff_out)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, lora_rank):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = LoRAAttention(dim, lora_rank)
        self.cross_attention = LoRAAttention(dim, lora_rank)
        self.feed_forward = LoRAFeedForward(dim, hidden_dim, lora_rank)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, enc_output):
        self_attn_out = self.self_attention(x)
        x = self.norm1(x + self_attn_out)
        cross_attn_out = self.cross_attention(x + enc_output)
        x = self.norm2(x + cross_attn_out)
        ff_out = self.feed_forward(x)
        return self.norm3(x + ff_out)

# 示例输入
batch_size = 2
seq_len = 10
dim = 32
hidden_dim = 64
lora_rank = 4

# 模拟输入数据
x = torch.randn(batch_size, seq_len, dim)
enc_output = torch.randn(batch_size, seq_len, dim)

# 实例化编码器和解码器层
encoder_layer = TransformerEncoderLayer(dim, hidden_dim, lora_rank)
decoder_layer = TransformerDecoderLayer(dim, hidden_dim, lora_rank)

# 前向传播
encoder_output = encoder_layer(x)
decoder_output = decoder_layer(x, encoder_output)

print("Encoder Output Shape:", encoder_output.shape)
print("Decoder Output Shape:", decoder_output.shape)
