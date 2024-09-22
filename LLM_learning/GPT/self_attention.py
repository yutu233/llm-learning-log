# 实现紧凑型self-attention类

import torch.nn as nn
import torch

class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        # 初始化可训练的权重矩阵
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # 前向传播
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vectors = attention_weights @ values
        return context_vectors
    
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        # 初始化可训练的权重矩阵
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        """
        参数解释:
        nn.Linear(d_in, d_out, bias=qkv_bias): 输入维度为d_in, 输出维度为d_out, 是否使用偏置项为qkv_bias
        nn.Linear(): 线性层，用于实现线性变换
        """

    def forward(self, x):
        # 前向传播
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vectors = attention_weights @ values
        return context_vectors
    