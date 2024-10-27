import torch
import torch.nn as nn

# 假设参数
embed_dim = 2560
num_heads = 8
num_layers = 3

# 模拟输入
input_embeddings = torch.rand(1, 21, embed_dim)  # torch.Size([1, 21, 2560])
projected_patch_embeddings = torch.rand(1, 729, embed_dim)  # torch.Size([1, 729, 2560])

# 定义多层交叉注意力模块
cross_attentions = nn.ModuleList([
    nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True) for _ in range(num_layers)
])

# # 逐层执行交叉注意力
# query = input_embeddings
# for layer, cross_attn in enumerate(cross_attentions):
#     # 确保 key 和 value 的形状与 batch_first=True 对应
#     attn_output, attn_weights = cross_attn(query, projected_patch_embeddings, projected_patch_embeddings)
#     print(f"Layer {layer+1} output shape: {attn_output.shape}")
#
#     query = attn_output  # 上一层的输出作为本层的查询

    # # 将交叉注意力输出与原始图片特征结合（残差连接）
    # projected_patch_embeddings = projected_patch_embeddings + attn_output


# 逐层执行交叉注意力
query = projected_patch_embeddings
for layer, cross_attn in enumerate(cross_attentions):
    # 确保 key 和 value 的形状与 batch_first=True 对应
    attn_output, attn_weights = cross_attn(query, input_embeddings, input_embeddings)
    print(f"Layer {layer+1} output shape: {attn_output.shape}")

    # 将交叉注意力输出与原始图片特征结合（残差连接）
    query = query + attn_output  # 更新 query 为残差连接结果

# 更新后的图片特征
projected_patch_embeddings = query

# 将图片特征与文本特征连接
multimodal_embeddings = torch.cat([projected_patch_embeddings, input_embeddings], dim=1)
