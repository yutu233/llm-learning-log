import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, emb_dim):
        """
        初始化函数

        Args:
            emb_dim (int): 嵌入向量的维度

        Returns:
            None

        """
        # 调用父类的初始化方法
        super().__init__()

        # 初始化一个小的正数，用于防止分母为0的情况
        self.eps = 1e-5

        # 初始化一个可学习的参数，用于缩放嵌入向量
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # 初始化一个可学习的参数，用于平移嵌入向量
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        对输入张量进行Batch Normalization操作。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, channels, *)，其中*表示任意维度。

        Returns:
            torch.Tensor: 进行Batch Normalization操作后的张量，形状与输入张量相同。

        """
        # 计算输入张量x在最后一个维度上的均值，并保持维度不变
        mean = x.mean(dim=-1, keepdim=True)
        # 计算输入张量x在最后一个维度上的方差，并保持维度不变，使用无偏估计
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 对输入张量x进行标准化处理，即(x - 均值) / sqrt(方差 + eps)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 对标准化后的张量进行缩放和平移，返回最终的Batch Normalization结果
        return self.scale * norm_x + self.shift