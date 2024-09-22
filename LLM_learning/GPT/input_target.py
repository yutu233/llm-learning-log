import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): # 返回数据集的长度
        return len(self.input_ids)
    def __getitem__(self, idx): # 返回数据集的idx位置的数据
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    """
    参数表:
    txt: 文本数据
    batch_size: 批大小
    max_length: 最大长度
    stride: 步长
    shuffle: 是否打乱
    drop_last: 是否丢弃最后一个不完整的batch
    """
    """
    函数作用:
    创建数据集和DataLoader
    返回值:
    DataLoader
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

with open("C:\VS CODE\LLM_learning\\resources\\the-verdict.txt", 
          "r", encoding="utf-8") as file:
    
    """raw_text: 读取txt文件
    dataloader: 创建DataLoader
    data_iter: 迭代器
    first_batch: 第一个batch
    iter(dataloader): 迭代器
    next(): 迭代器"""
    raw_text = file.read()
    # dataloader = create_dataloader_v1(
        # raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # print(first_batch)
    # second_batch = next(data_iter)
    # print(second_batch)
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print(f"Inputs:\n{inputs}")
    # print(f"\nTargets:\n{targets}")

# 仿照文件embedding_simple.py, 创建一个初始位置编码
output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)

# 用token_embedding_layer将这些token ID 嵌入为256维向量
token_embeddings = token_embedding_layer(inputs)

# print(token_embeddings.shape)

# 创建一个与token_embeddings形状相同的position_embeddings嵌入层
context_length = max_length
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)
position_embeddings = position_embedding_layer(torch.arange(context_length))

# print(position_embeddings.shape)

# 将position_embeddings添加到嵌入标记中
input_embeddings = token_embeddings + position_embeddings

# print(input_embeddings.shape)