import torch
import torch.nn as nn
from torch.nn import LayerNorm
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(self, config):
        """
        初始化函数，用于创建Transformer编码器层的实例。

        Args:
            config (dict): 包含模型配置信息的字典，包括：
                - embedding_dim (int): 嵌入层维度。
                - context_length (int): 上下文长度。
                - num_heads (int): 多头自注意力机制中的头数。
                - drop_rate (float): Dropout层的丢弃率。
                - qkv_bias (bool): 是否在多头自注意力机制的Q、K、V变换矩阵中启用偏置项。

        Returns:
            None

        """
        super().__init__()
        # 创建多头自注意力机制层
        self.att = MultiHeadAttention(
            # 输入维度
            d_in=config["embedding_dim"],
            # 输出维度
            d_out=config["embedding_dim"],
            # 上下文长度
            context_length=config["context_length"],
            # 头数
            num_heads=config["num_heads"],
            # Dropout层的丢弃率
            dropout=config["drop_rate"],
            # 是否在Q、K、V变换矩阵中启用偏置项
            qkv_bias=config["qkv_bias"],
        )
        # 创建前馈神经网络层
        self.ff = FeedForward(config)
        # 创建第一层LayerNorm层
        self.norm1 = LayerNorm(config["embedding_dim"])
        # 创建第二层LayerNorm层
        self.norm2 = LayerNorm(config["embedding_dim"])
        # 创建Dropout层
        self.drop_resid = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_channels, height, width)
    
        Returns:
            torch.Tensor: 输出张量，形状与输入张量相同
    
        """
        # 保留原始输入张量作为shortcut
        shortcut = x
        # 对输入张量进行归一化处理
        x = self.norm1(x)
        # 对归一化后的张量进行注意力处理
        x = self.att(x)
        # 对注意力处理后的张量进行残差丢弃处理
        x = self.drop_resid(x)
        # 将原始输入张量shortcut与残差丢弃后的张量相加
        x = x + shortcut

        # 更新shortcut为上一轮的输出结果
        shortcut = x
        # 对上一轮的输出结果进行归一化处理
        x = self.norm2(x)
        # 对归一化后的张量进行全连接处理
        x = self.ff(x)
        # 对全连接处理后的张量进行残差丢弃处理
        x = self.drop_resid(x)
        # 将上一轮的shortcut与残差丢弃后的张量相加
        x = x + shortcut

        return x
    
GPT_CONFIG_124M = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 256, # Context length
    "embedding_dim": 768, # Embedding dimension
    "num_heads": 12, # Number of attention heads
    "num_layers": 12, # Number of layers
    "drop_rate": 0.1, # Dropout rate
    "qkv_bias": False # Query-Key-Value bias
}