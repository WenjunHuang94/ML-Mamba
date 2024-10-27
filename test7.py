import torch
import torch.nn as nn

class PerceptualWeightGenerator(nn.Module):
    def __init__(self, visual_dim, hidden_dim, lora_rank, num_common_features, num_specific_features):
        super(PerceptualWeightGenerator, self).__init__()
        self.common_features_layer = nn.Linear(visual_dim, num_common_features)
        self.specific_features_layer = nn.Linear(visual_dim, num_specific_features)
        self.lora_A_common = nn.Linear(num_common_features, lora_rank)
        self.lora_B_common = nn.Linear(lora_rank, visual_dim)
        self.lora_A_specific = nn.Linear(num_specific_features, lora_rank)
        self.lora_B_specific = nn.Linear(lora_rank, visual_dim)

    def forward(self, visual_features):
        common_features = self.common_features_layer(visual_features)
        common_weights = self.lora_A_common(common_features) @ self.lora_B_common.weight.t()

        specific_features = self.specific_features_layer(visual_features)
        specific_weights = self.lora_A_specific(specific_features) @ self.lora_B_specific.weight.t()

        perceptual_weights = common_weights + specific_weights
        return perceptual_weights

# 示例参数设置
visual_dim = 768
hidden_dim = 512
lora_rank = 16
num_common_features = 256
num_specific_features = 256

# 实例化感知权重生成器
pwg = PerceptualWeightGenerator(visual_dim, hidden_dim, lora_rank, num_common_features, num_specific_features)

# 示例输入
visual_features = torch.rand(1, 10, visual_dim)  # 假设有一批次的视觉特征
perceptual_weights = pwg(visual_features)
print(perceptual_weights.shape)
