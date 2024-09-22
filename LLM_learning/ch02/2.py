import torch

# 假设有三个训练样本
# 这些样本可能表示语言模型(LM)上下文中的标记ID
idx = torch.tensor([2, 3, 1])
# 嵌入矩阵的行数通过获取最大标记ID + 1来确定
# 如果最高的标记ID是3, 则我们希望有4行, 对应可能的标记0, 1, 2, 3
num_idx = max(idx) + 1

# 所需的嵌入维度是一个超参数
out_dim = 5

# 实现一个简单的嵌入层
torch.manual_seed(123)

# 创建一个嵌入层, 指定输入维度为num_idx, 输出维度为out_dim
embedding = torch.nn.Embedding(num_idx, out_dim)

# 查看嵌入权重数据情况
# print(embedding.weight)

# 使用嵌入层来获取具有ID 1的训练样本的向量表示
# print(embedding(torch.tensor([1])))
# 可视化
# 1) Index of training example [1]
# 2) Embeddind matrix
# 将原先的第三行变成现在的第一行，第四行变成第二行，第二行变成第三行
idx = torch.tensor([2, 3, 1])
# print(embedding(idx))

# 使用One-Hot编码
onehot = torch.nn.functional.one_hot(idx)
"""
在Pytorch中, one-hot函数用于将给定索引转换为独热编码张量
其中每个类别被表示为一个仅在对应类别位置为1, 其余位置为0的向量
torch.nn.functional.one_hot(idx, num_classes=None):
    idx: 一个包含类别索引的张量, 索引可以是多维的, 但最后一个维度会被视为索引维度
    num_classes(可选): 指定独热编码中类别的总数, 如果未指定, 则在最后一个维度上扩展为num_classes大小的独热编码张量

"""
# print(onehot)

# 使用矩阵乘法XW.T来初始化一个Linear层
torch.manual_seed(123)
# 初始化一个Linear层, 该层的权重是由num_idx(输入维度)到out_dim(输出维度)的一个线性层, 且没有偏置
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
# print(linear.weight)
# 注意, Pytorch中的Linear层也是用很小的随机权重初始化
# Linear层的权重被重新赋值为与Embedding层相同的权重
# 确保它们具有相同的初始化
linear.weight = torch.nn.Parameter(embedding.weight.T.detach())
"""
torch.nn.Parameter:
    是一个特殊的Tensor, 当它被赋给一个Module的属性时, 它会被自动注册为参数
    这意味着Pytorch会在反向传播时自动计算它的梯度
    并且在训练时更新它们的值
.T:
    是矩阵转置的操作符
    它将矩阵的行和列互换
.detach():
    从计算图中分离出Tensor, 返回一个新的Tensor
    这个新Tensor与原Tensor共享数据内存
    但不再要求梯度, 即它不再是一个计算图的子节点
    这通常用于防止梯度通过某个路径回流
linear:
    Linear层执行线性变换
    即y = x * W.T + b
    其中x为输入, W为权重矩阵, b为偏置项, y为输出
"""

# print(linear(onehot.float()))