# 该文件包含训练模型所需的大部分模块和函数

# 导入必要的库
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# torch: 用于构建和训练神经网络
# torch.nn: 用于构建神经网络
# torch.utils.data: 用于构建数据集
# tik token: 用于生成随机字符串

class GPTDatasetV1(Dataset): # 定义数据集类
    # 定义数据集类，用于读取数据集
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        初始化一个实例，用于生成输入序列和输出序列
        
        Args:
            txt (str): 待编码的文本
            tokenizer (object): 用于编码文本的tokenizer
            max_length (int): 输入序列的最大长度
            stride (int): 遍历文本时，每次取max_length个token作为输入，并预测第max_length+1个token作为输出的步长
        
        Returns:
            None
        
        """
        self.tokenizer = tokenizer # 传入tokenizer
        self.input_ids = [] # 输入序列
        self.target_ids = [] # 输出序列

        token_ids = tokenizer.encode(txt) # 编码文本

        for i in range(0, len(token_ids) - max_length, stride): # 遍历文本，每次取max_length个token作为输入，并预测第max_length+1个token作为输出
            input_chunk = token_ids[i:i + max_length] # 取输入序列
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            # range(start, stop, step):内置函数, 用于生成一个数字列表, 从start(包含)到stop(不包含)的整数, 每隔step取一个数
            # 在上面的循环中, start为0, stop是len(token_ids) - max_length, step为stride, 所以range函数会遍历文本中的所有可能的输入序列
            # 之所以是len(token_ids) - max_length, 是因为需要确保第max_length + 1可以被预测

    def __len__(self):
        """
        当调用len()方法时, 会自动调用该函数

        Returns:
                len(self.input_ids):int: 返回编码的输入token
        """
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        """
        根据索引获取输入ID和目标ID。
        
        Args:
            idx (int): 输入数据的索引。
        
        Returns:
            tuple: 一个元组，包含两个元素：
            - 输入ID（list）：根据索引从self.inputids中获取的元素。
            - 目标ID（list）：根据索引从self.target_ids中获取的元素。
        
        """
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True):
    """
    根据给定的txt文件路径创建GPT模型训练的数据加载器
    
    Args:
        txt (str): txt文件路径
        batch_size (int, optional): 每个批次的数据量. Defaults to 4.
        max_length (int, optional): 输入文本的最大长度. Defaults to 256.
        stride (int, optional): 采样步长. Defaults to 128.
        shuffle (bool, optional): 是否在每次迭代时打乱数据. Defaults to True.
        drop_last (bool, optional): 是否丢弃最后一个不完整的批次. Defaults to True.
    
    Returns:
        DataLoader: 数据加载器
    
    """
    # 获取GPT-2模型的分词器
    tokenizer = tiktoken.get_encoding('gpt2')
    # 创建数据集实例, 将文本转化为适合模型训练的格式
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # 创建数据加载器, 负责在训练过程中以批次形式提供数据
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    # 返回创建好的DataLoader实例
    return dataloader

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        """
        初始化MultiHeadAttention类。
        
        Args:
            d_in (int): 输入特征的维度。
            d_out (int): 输出特征的维度。
            block_size (int): 自注意力机制的序列长度，用于生成注意力掩码，以确保模型在训练时不会看到未来的信息。
            dropout (float): Dropout层的概率值。
            num_heads (int): 自注意力机制中头部的数量。
            qkv_bias (bool, optional): 查询、键和值线性层的偏置项。默认为False。
        
        Returns:
            None
        
        Raises:
            AssertionError: 如果d_out不能被num_heads整除，则抛出异常。
        
        """
        super().__init__()
        # 断言检查，确保d_out能被num_heads整除，因为每个头的输出维度应该是相同的，确保它们可以被拼接成最终的输出
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"
        # 初始化实例变量
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # 线性层
        # 创建三个线性层，分别生成Q,K向量，V向量通常直接来自输出
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # 注意力掩码
        # 使用torch.ones创建注意力掩码，并将其注册为缓冲区
        # 基于block_size生成，这是一个上三角矩阵，对角线以上的元素为1(表示需要被屏蔽的位置)
        # 但triu默认包含对角线元素
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))
        # 缓冲区是那些不需要梯度的持久化张量，他们通常用于存储模型在训练过程中不会改变的参数
        """
        self.register_buffer(...):这是nn.Module类中的一个方法
        用于注册一个缓冲区
        'mask': 缓冲区的名称(str)
        torch.triu(): 要注册为缓冲区的张量
        """