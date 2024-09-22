import torch
from torch import nn

class GELU(nn.Module):

    def __init__(self):
        # 调用父类的构造函数
        super().__init__()

    def forward(self, x):
        # 使用tanh函数和x的线性组合计算输出
        return 0.5 * x * (1 + torch.tanh(
            # 计算sqrt(2/pi) * (x + 0.044715 * x^3)
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))