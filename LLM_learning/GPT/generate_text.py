import torch
import tiktoken
from train_example import *
from final_encode import GPTModel

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    根据给定的模型和上下文生成一段文本。
    
    Args:
        model (torch.nn.Module): 用于生成文本的模型。
        idx (torch.Tensor): 输入的文本序列，形状为 (batch_size, seq_len)。
        max_new_tokens (int): 最多生成多少个新token。
        context_size (int): 模型生成每个新token时考虑的上下文长度。
    
    Returns:
        torch.Tensor: 生成的文本序列，形状为 (batch_size, seq_len + max_new_tokens)。
    
    """
    for _ in range(max_new_tokens):
        # 获取当前需要作为条件的上下文序列
        idx_cond = idx[:, -context_size:]

        # 关闭梯度计算，因为我们只是进行推断，不需要反向传播
        with torch.no_grad():
            # 将上下文序列传入模型，得到预测结果
            logits = model(idx_cond)

        # 取出最后一个时间步的预测结果
        logits = logits[:, -1, :]

        # 对预测结果应用softmax函数，得到概率分布
        probas = torch.softmax(logits, dim=-1)

        # 选择概率最大的token作为下一个token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 将下一个token拼接到当前的文本序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    """
    将文本转换为token id的tensor。

    Args:
        text (str): 待转换的文本。
        tokenizer (PreTrainedTokenizer): 用于编码文本的tokenizer。

    Returns:
        torch.Tensor: 转换后的token id的tensor，shape为(1, n)，其中n为token数量。

    """
    # 使用tokenizer对文本进行编码，并允许特定的特殊符号'<|endoftext|>'
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 将编码后的token id列表转换为tensor，并在第一个维度上增加一个维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    将token_ids转换为文本

    Args:
        token_ids (torch.Tensor): 形状为[batch_size, sequence_length]的token id张量
        tokenizer (PreTrainedTokenizer): 使用的tokenizer对象

    Returns:
        str: 转换后的文本

    """
    # 将token_ids从二维张量压缩为一维张量
    flat = token_ids.squeeze(0)
    # 使用tokenizer的decode方法将token id列表解码为文本
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                       [40, 1107, 588]]) # "I really like"]
targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                        [588, 428, 11311]]) # " really like chocolate"]

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)

# print(probas.shape)

# 该行代码用于获取概率分布中每个样本的最大索引
# 返回的 token_ids 是一个张量，表示每个样本的预测类别
token_ids = torch.argmax(probas, dim=-1, keepdim=True)

# print("Token IDs:\n", token_ids)
# print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
# print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
# 从概率矩阵 probas 中提取目标类别的概率
# text_idx 是当前文本的索引，targets 是目标类索引
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]

# print("Text1: ", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]

#print("Text2: ", target_probas_2)

# 计算目标概率的对数
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))

# print(log_probas)

# 计算对数概率的平均值
avg_log_probas = torch.mean(log_probas)

# print(avg_log_probas)

# 取平均对数概率的负值
neg_avg_log_probas = avg_log_probas * -1

# print(neg_avg_log_probas)

# print("Logits shape:", logits.shape)
# print("Targets shape:", targets.shape)

# 将logits和targets展开为一维
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()

# print("Flattened logits:", logits_flat.shape)
# print("Flattened targets:", targets_flat.shape)

# 计算交叉熵损失
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

# print(loss)

# 计算困惑度
perplexity = torch.exp(loss)

# print(perplexity)

# 计算训练集和验证集损失