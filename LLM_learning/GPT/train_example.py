import torch
import torch.nn as nn
import tiktoken
from final_encode import *

# test
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

# 将文本编码为张量并添加到批处理中
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
# 将批处理中的张量堆叠在一起
batch = torch.stack(batch, dim=0)

out = model(batch)

# print("Input batch:\n", batch)
# print("\nOutput shape:", out.shape)
# print(out)

# total_params = sum(p.numel() for p in model.parameters())

# print(f"Total number of parameters: {total_params}")

# total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
# print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
# total_size_bytes = total_params * 4

# # Convert to megabytes
# total_size_mb = total_size_bytes / (1024 * 1024)

# print(f"Total size of the model: {total_size_mb:.2f} MB")
