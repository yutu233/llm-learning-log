import torch
import torch.nn as nn
from train_example import tokenizer, GPT_CONFIG_124M, model, GPTModel
from input_target import create_dataloader_v1
from generate_text import text_to_token_ids, generate_text_simple, token_ids_to_text
import matplotlib.pyplot as plt
import tiktoken

file_path = "GPT\\the-verdict.txt"

# 加载数据集
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# 检查字符数和token数
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

# print(f"Characters: {total_characters}")
# print(f"Tokens: {total_tokens}")

# 数据分割和加载
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True
)

# 检查
# print("Train loader:")
# for x, y in train_loader:
#     print(x.shape, y.shape)
# print("\nValidation loader:")
# for x, y in val_loader:
#     print(x.shape, y.shape)

# 计算训练和验证加载器返回的特定批次的交叉熵损失
def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算一个批次数据的交叉熵损失。

    Args:
        input_batch (torch.Tensor): 输入数据的批次张量。
        target_batch (torch.Tensor): 目标数据的批次张量。
        model (torch.nn.Module): 用于计算logits的模型。
        device (str): 设备类型，'cuda' 或 'cpu'。

    Returns:
        torch.Tensor: 交叉熵损失。
    """
    # 将输入数据和目标数据转移到指定设备上
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 通过模型计算logits
    logits = model(input_batch)

    # 计算交叉熵损失
    # 将logits和目标数据展平
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算给定模型在给定数据加载器上的平均损失值。

    Args:
        data_loader (torch.utils.data.DataLoader): 用于加载数据的数据加载器。
        model (torch.nn.Module): 用于计算损失值的模型。
        device (torch.device): 用于计算损失值的设备（CPU 或 GPU）。
        num_batches (int, optional): 要计算损失值的批次数量。默认为 None，即计算所有数据加载器中的批次。

    Returns:
        float: 给定模型在给定数据加载器上的平均损失值。

    """
    total_loss = 0.
    if num_batches is None:
        # 如果未指定 num_batches，则计算所有数据加载器中的批次
        num_batches = len(data_loader)
    else:
        # 否则，取 num_batches 和数据加载器中的批次数量的较小值
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算当前批次的损失值
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 累加总损失值
            total_loss += loss.item()
        else:
            # 如果已计算完指定数量的批次，则跳出循环
            break
    # 返回平均损失值
    return total_loss / num_batches

# 设置设备为CUDA（如果可用），否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型移至指定设备
model.to(device)
train_loss = calc_loss_loader(train_loader, model, device)
val_loss = calc_loss_loader(val_loader, model, device)

# print("Training Loss:", train_loss)
# print("Validation Loss:", val_loss)

# 训练模型
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    """
    训练模型。
    
    Args:
        model: 用于训练的模型。
        train_loader: 训练数据的DataLoader。
        val_loader: 验证数据的DataLoader。
        optimizer: 优化器。
        device: 设备，可以是'cpu'或'cuda'。
        num_epochs: 训练轮数。
        eval_freq: 每多少个训练步骤进行一次验证。
        eval_iter: 用于验证的迭代次数。
        start_context: 生成样本时使用的初始上下文。
    
    Returns:
        一个包含训练损失、验证损失和跟踪的tokens数量的列表。
    
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 计算损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 反向传播
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累加已处理的tokens数量
            tokens_seen += input_batch.numel()
            # 全局步骤数递增
            global_step += 1

            # 如果达到验证频率，则进行验证
            if global_step % eval_freq == 0:
                # 验证模型，并返回训练损失和验证损失
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                # 添加训练损失和验证损失到列表中
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                # 添加已处理的tokens数量到列表中
                track_tokens_seen.append(tokens_seen)
                # 打印训练损失和验证损失
                print(f"Ep {epoch + 1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 生成并打印样本
        generate_and_print_sample(
            model, train_loader.dataset.tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的性能。

    Args:
        model (torch.nn.Module): 待评估的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证数据加载器。
        device (torch.device): 计算设备。
        eval_iter (int): 评估时使用的迭代次数。

    Returns:
        Tuple[float, float]: 包含训练集和验证集上损失值的元组。

    """
    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        # 计算训练集上的损失值
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # 计算验证集上的损失值
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 设置模型为训练模式
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    生成并打印样本文本。
    
    Args:
        model (torch.nn.Module): 预训练的语言模型。
        tokenizer (transformers.PreTrainedTokenizer): 用于编码和解码文本的tokenizer。
        device (torch.device): 用于计算的设备。
        start_context (str): 用于生成文本的起始上下文。
    
    Returns:
        None
    """
    # 将模型设置为评估模式
    model.eval()

    # 获取模型位置嵌入的权重形状的第一个维度，即上下文大小
    context_size = model.pos_emb.weight.shape[0]

    # 将起始上下文编码为token id，并转移到指定设备
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        # 生成文本
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )

        # 将生成的token id解码为文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)

        # 打印解码后的文本，将换行符替换为空格
        print(decoded_text.replace("\n", " "))

    # 将模型设置回训练模式
    model.train()

# an exanple
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, track_tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,
    eval_freq=5, eval_iter=1, start_context="Every effort moves you"
)

# 绘制损失值和tokens数量的曲线
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    绘制训练损失和验证损失随训练轮数变化的折线图。

    Args:
        epochs_seen (list): 训练轮数列表。
        tokens_seen (list): 已经看过的tokens数量列表。
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。

    Returns:
        None

    """
    # 创建一个5x3大小的图形窗口
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # 在第一个坐标轴上绘制训练损失随训练轮数变化的折线图
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    # 在第一个坐标轴上绘制验证损失随训练轮数变化的折线图
    ax1.plot(epochs_seen, val_losses, linestyle="-", label="Validation loss")
    # 设置第一个坐标轴的x轴标签为"Epochs"
    ax1.set_xlabel("Epochs")
    # 设置第一个坐标轴的y轴标签为"Loss"
    ax1.set_ylabel("Loss")
    # 在第一个坐标轴上的右上角显示图例
    ax1.legend(loc="upper right")
    # 创建与第一个坐标轴共享y轴的第二个坐标轴
    ax2 = ax1.twiny()
    # 在第二个坐标轴上绘制训练损失随已看过的tokens数量变化的折线图，但设置透明度为0，使其不显示
    ax2.plot(tokens_seen, train_losses, alpha=0)
    # 设置第二个坐标轴的x轴标签为"Tokens seen"
    ax2.set_xlabel("Tokens seen")
    # 调整图形布局，避免标签重叠
    fig.tight_layout()
    # 显示图形窗口
    plt.show()

model.to('cpu')
model.eval()

tokenizer = tiktoken.get_encoding('gpt2')
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

# print("Outout text:\n", token_ids_to_text(token_ids, tokenizer))

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()

# print(inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()

# print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    """
    根据给定的概率分布进行采样，并打印每个token的采样频率。
    
    Args:
        probas (torch.Tensor): 形状为 (vocab_size,) 的概率分布张量，其中 vocab_size 为词汇表大小。
    
    Returns:
        None
    
    """
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1) for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature):
    """
    对输入logits进行带温度的softmax操作。
    
    Args:
        logits (torch.Tensor): 输入的未归一化的概率分布，shape为[num_classes]。
        temperature (float): 温度系数，用于控制softmax的平滑程度。当temperature趋近于0时，softmax输出趋近于one-hot向量；
                            当temperature趋近于无穷大时，softmax输出趋近于均匀分布。
    
    Returns:
        torch.Tensor: 经过softmax归一化后的概率分布，shape为[num_classes]。
    
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# temperatures = [1, 0.1, 5]# 初始, 更低, 更高的温度
# scaled_probas = [
#     softmax_with_temperature(next_token_logits, T) for T in temperatures
# ]
# x = torch.arange(len(vocab))
# bar_width = 0.15
# fig, ax = plt.subplots(figsize=(5, 3))
# for i, T in enumerate(temperatures):
#     rects = ax.bar(x + i * bar_width, scaled_probas[i],
#                    bar_width, label=f'Temperature = {T}')
#     ax.set_ylabel('Probability')
#     ax.set_xticks(x)
#     ax.set_xticklabels(vocab.keys(), rotation=90)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

top_k = 3
top_logits, top_pos = torch.topk(
    next_token_logits, top_k
)

# print("Top logits:", top_logits)
# print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float('-inf')),
    other=next_token_logits
)

# print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)

# print(topk_probas)

def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):
    """
    根据给定的模型和上下文生成新的文本。
    
    Args:
        model (torch.nn.Module): 用于生成文本的模型。
        idx (torch.Tensor): 初始的文本索引张量，形状为 (batch_size, seq_len)。
        max_new_tokens (int): 生成的最大新token数量。
        context_size (int): 用于生成下一个token的上下文大小。
        temperature (float): 控制生成文本的随机性，温度越高，生成的文本越随机。
        top_k (Optional[int]): 用于过滤logits的top k值，如果为None则不进行过滤。
    
    Returns:
        torch.Tensor: 生成的新文本索引张量，形状为 (batch_size, seq_len + max_new_tokens)。
    
    """
    for _ in range(max_new_tokens):
        # 获取用于生成下一个token的上下文
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # 获取模型预测的logits
            logits = model(idx_cond)
        # 取最后一个位置的logits
        logits = logits[:, -1, :]

        # 如果提供了top_k值，则进行过滤
        if top_k is not None:
            # 获取logits中最大的top_k个值
            top_logits, _ = torch.topk(logits, top_k)
            # 获取top_k中最小的值
            min_val = top_logits[:, -1]
            # 将logits中小于min_val的值替换为-inf
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(
                    logits.device
                ),
                logits
            )

        # 根据温度调整logits
        if temperature > 0.0:
            logits = logits / temperature
            # 将logits转换为概率分布
            probas = torch.softmax(logits, dim=-1)
            # 根据概率分布进行采样得到下一个token的索引
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            # 直接选择概率最大的token作为下一个token的索引
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 将生成的下一个token的索引拼接到原索引张量后面
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
    
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 保存模型权重
torch.save(model.state_dict(), "model_pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model_pth"))
model.eval()

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    },
    'model_and_optimizer.pth'
)
checkpoint = torch.load('model_and_optimizer.pth')
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.train()