import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRA, self).__init__()
        self.A = nn.Parameter(torch.randn(out_features, r))
        self.B = nn.Parameter(torch.randn(r, in_features))

    def forward(self, x):
        return x @ self.B.T @ self.A.T


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, r: int, num_specific: int,
                 mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.linear1 = nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True)
            self.linear2 = nn.Linear(self.initial_projection_dim, llm_dim, bias=True)
            self.linear3 = nn.Linear(llm_dim, llm_dim, bias=True)
            self.gelu = nn.GELU()

            # 公共LoRA模块
            self.shared_lora = LoRA(llm_dim, llm_dim, r)

            # 多个特定LoRA模块
            self.specific_loras = nn.ModuleList([LoRA(llm_dim, llm_dim, r) for _ in range(num_specific)])
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, x):
        # 原始线性层
        x = self.gelu(self.linear1(x))
        x = self.gelu(self.linear2(x))
        x = self.linear3(x)

        # 计算共享LoRA模块的输出
        shared_output = self.shared_lora(x)

        # 计算所有特定LoRA模块的输出，并与共享输出相加
        specific_outputs = [specific_lora(x) + shared_output for specific_lora in self.specific_loras]

        # 将所有特定LoRA模块的输出结合起来（例如求和）
        combined_output = sum(specific_outputs)

        # 最终输出
        x = x + combined_output

        return x


# 冻结原始参数，只更新LoRA参数
def freeze_original_params(model):
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False



# 示例输入数据
batch_size = 1
sequence_length = 729
feature_dimension = 2560

# 创建随机输入数据
projected_patch_embeddings = torch.randn(batch_size, sequence_length, feature_dimension)

# 初始化FusedMLPProjector
fused_vision_dim = feature_dimension
llm_dim = feature_dimension  # 输出维度与输入维度相同
r = 16  # LoRA的秩
num_specific = 3  # 特定LoRA模块的数量

# 创建FusedMLPProjector实例
fused_mlp_projector = FusedMLPProjector(fused_vision_dim, llm_dim, r, num_specific)

# 运行模型
output = fused_mlp_projector(projected_patch_embeddings)

# 打印输出的形状
print("Output shape:", output.shape)
