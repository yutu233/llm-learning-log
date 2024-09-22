# 因果注意力，又称遮蔽注意力，是自注意力的一种特殊形式
# 它通过对输入序列的每个元素进行注意力权重的计算，来对输入序列进行筛选，只保留与输出相关的元素，并将其余元素遮蔽掉
import torch
from self_attention import *
"""
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts (x^3)
     [0.22, 0.58, 0.33], # with (x^4)
     [0.77, 0.25, 0.10], # one (x^5)
     [0.05, 0.80, 0.55]] # step (x^6)
)
d_in = inputs.shape[1]
d_out = 2
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=1)

print(attention_weights)

# 创建一个遮蔽, 使对角线以上的值为零:
context_length = attention_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))

print(mask_simple)

"""
"""代码解释:
torch.tril(): 返回一个下三角矩阵，即矩阵的对角线以上的元素都为零。
torch.ones(): 返回一个全1的矩阵。
"""
"""

masked_simple = attention_weights * mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_normalized = masked_simple / row_sums

print(masked_simple_normalized)

""""""
代码解释:
masked_simple: 遮蔽后的注意力权重。
row_sums: 对每一行求和，得到每一行的归一化因子。
sum(dim=1, keepdim=True): 对每一行求和，得到一个一维向量。
dim=1: 按行求和。
keepdim=True: 保持维度。
masked_simple_normal: 归一化后的遮蔽后的注意力权重。
""""""

# 优化
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)

print(masked)

""""""
代码解释:
torch.triu(): 返回一个上三角矩阵，即矩阵的对角线以下的元素都为零。
diagonal=1: 取对角线以下的元素。
masked_fill(): 用-inf填充矩阵中满足条件的元素。
mask.bool(): 将mask转换为布尔值。
torch.inf: 无穷大。
""""""

attention_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attention_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)

print(dropout(example))
print(dropout(attention_weights))

batch = torch.stack((inputs, inputs), dim=0)

print(batch.shape)

""""""
torch.stack(): 将多个张量堆叠成一个新的张量。
(inputs, inputs): 两个相同的输入张量。
dim=0: 按行堆叠。
""""""
# torch.Size([2, 6, 3]): 2个输入文本, 每个文本有6个token, 每个token是三维嵌入向量
"""

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2) # 改变transpose
        attention_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector
    
    """
    mask.bool()[:num_tokens, :num_tokens]: 取上三角矩阵的上三角部分。
    """

"""
# an example
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vectors = ca(batch)

# print("context_vectors.shape: ", context_vectors.shape)
"""