import torch
import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, config):
        """
        初始化函数，用于构建模型。

        Args:
            config (dict): 包含模型配置信息的字典，其中必须包含"embedding_dim"字段。

        Returns:
            None
        """
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个线性层，输入维度为config["embedding_dim"]，输出维度也为config["embedding_dim"]
        self.linear1 = nn.Linear(config["embedding_dim"], config["embedding_dim"] * 4)

        # 创建一个ReLU激活函数
        self.relu = nn.ReLU()

        # 创建一个线性层，输入维度为config["embedding_dim"] * 4，输出维度为config["embedding_dim"]
        self.linear2 = nn.Linear(config["embedding_dim"] * 4, config["embedding_dim"])

        # 创建一个Dropout层，丢弃率为config["drop_rate"]
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # 将输入x通过线性层self.linear1进行线性变换，并通过relu激活函数
        x = self.relu(self.linear1(x))
        # 对经过relu激活函数后的x进行dropout操作
        x = self.dropout(x)
        # 将dropout后的x通过线性层self.linear2进行线性变换
        x = self.linear2(x)
        # 返回经过线性变换的x
        return x