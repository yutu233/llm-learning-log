from torch import nn
from causal_attention import *

class MultiHeadAttentionWrapper(nn.Module):
    
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        """
        初始化多头自注意力机制的模型对象
    
        Args:
            d_in (int): 输入特征的维度大小
            d_out (int): 输出特征的维度大小
            context_length (int): 上下文长度
            dropout (float): dropout率
            num_heads (int): 自注意力机制的头数
            qkv_bias (bool, optional): 是否在Q, K, V的线性变换中添加偏置项. Defaults to False.
    
        Returns:
            None
    
        """
        super().__init__()
        # 创建一个用于存储多个自注意力机制的模块的列表
        # 创建一个模块列表，用于存储多个自注意力机制的模块
        self.heads = nn.ModuleList(
            # 使用列表推导式，根据头数num_heads生成多个CausalAttention模块，并添加到列表中
            # 创建一个列表推导式，根据头数num_heads生成多个CausalAttention模块
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        """
        将多个head的输出在最后一个维度上拼接起来。

        Args:
            x (torch.Tensor): 输入的tensor，shape为(batch_size, *, in_channels)。

        Returns:
            torch.Tensor: 拼接后的tensor，shape为(batch_size, *, num_heads * out_channels)。

        """
        # 遍历self.heads中的每个head
        # 对每个head执行head(x)，得到输出
        # 将所有head的输出在最后一个维度上拼接起来
        return torch.cat([head(x) for head in self.heads], dim=-1)

"""
# an example
torch.manual_seed(123)
context_length = batch.shape[1] # 这是token的数量
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vectors = mha(batch) # (batch_size, *, num_heads * out_channels)

print(context_vectors)
print("context_vectors.shape: ", context_vectors.shape)"""

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头自注意力模块。

        Args:
            d_in (int): 输入特征的维度大小。
            d_out (int): 输出特征的维度大小。
            context_length (int): 上下文序列的长度。
            dropout (float): dropout层的丢弃率。
            num_heads (int): 多头注意力的头数。
            qkv_bias (bool, optional): 查询、键、值线性层是否使用偏置。默认为False。

        Returns:
            None

        """
        super().__init__()
        # 在多头注意力机制中，确保输出维度可以被头数整除的断言
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        # 输出特征的维度大小
        self.d_out = d_out
        # 多头注意力的头数
        self.num_heads = num_heads
        # 每个头对应的维度大小
        self.head_dim = d_out // num_heads
        # 查询线性层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 键线性层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 值线性层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 输出投影线性层
        self.out_projection = nn.Linear(d_out, d_out)
        # dropout层
        self.dropout = nn.Dropout(dropout)
        # 创建一个掩码张量，用于多头自注意力中的上三角矩阵掩码
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, num_tokens, d_in)。

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, num_tokens, d_out)。

        """
        # 获取输入张量的形状
        b, num_tokens, d_in = x.shape

        # 将输入张量通过不同的线性层得到键、查询和值
        # 获取键
        keys = self.W_key(x)
        # 获取查询
        queries = self.W_query(x)
        # 获取值
        values = self.W_value(x)

        # 改变张量的形状以适应多头注意力机制
        # 改变键的形状
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        # 改变查询的形状
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # 改变值的形状
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 交换张量的维度，使得每个头对应的键、查询和值可以单独处理
        # 交换键的维度
        keys = keys.transpose(1, 2)
        # 交换查询的维度
        queries = queries.transpose(1, 2)
        # 交换值的维度
        values = values.transpose(1, 2)

        # 计算注意力分数
        # 计算查询和键的点积
        attention_scores = queries @ keys.transpose(2, 3)

        # 获取掩码矩阵的布尔值
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 将掩码位置的注意力分数填充为负无穷大
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # 对注意力分数进行softmax计算，得到注意力权重
        # 对注意力分数进行缩放后应用softmax函数
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        # 应用dropout层
        attention_weights = self.dropout(attention_weights)
    
        # 计算上下文向量
        # 计算注意力权重和值的加权和
        context_vector = (attention_weights @ values).transpose(1, 2)

        # 没有线性层，直接调整形状得到最终的输出张量
        # 将上下文向量调整形状为 (batch_size, num_tokens, d_out)
        # 没有线性层得到最终的输出张量
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)

        # 计算上下文向量
        # 应用输出投影层
        context_vector = self.out_projection(context_vector)

        return context_vector

"""
# an example
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], #A
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
 
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

# print(a @ a.transpose(2, 3))

# 取出第一行第一列的元素
first_head = a[0, 0, :, :]
# 计算第一行第一列的元素与其自身的转置矩阵乘积
first_result = first_head @ first_head.T

# 打印第一行第一列的元素与其自身的转置矩阵乘积
# print("First head:\n", first_result)

# 取出第一行第二列的元素
second_head = a[0, 1, :, :]
# 计算第一行第二列的元素与其自身的转置矩阵乘积
second_result = second_head @ second_head.T

# 打印第一行第二列的元素与其自身的转置矩阵乘积
# print("\nSecond head:\n", second_result)

# 设置随机种子
torch.manual_seed(123)
# 获取batch的形状
batch_size, context_length, d_in = batch.shape
# 定义输出维度
d_out = 2
# 实例化MultiHeadAttention类
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
# 传入batch，获取注意力机制的上下文向量
context_vectors = mha(batch)

# 打印上下文向量
# print(context_vectors)
# 打印上下文向量的形状
# print("context_vectors.shape: ", context_vectors.shape)"""