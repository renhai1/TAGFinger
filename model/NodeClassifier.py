import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeClassifier(nn.Module):
    """
    两层 MLP 分类头
    h  ->  ReLU -> Dropout -> hidden -> ReLU -> logits
    """

    def __init__(self,
                 hid_dim: int,  # 输入维度 = GNN/Prompt 输出维度
                 num_classes: int,  # 类别数
                 inner_dim: int = 128,  # 中间层维度，可调
                 dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(hid_dim, inner_dim)
        self.lin2 = nn.Linear(inner_dim, num_classes)
        self.dropout = dropout

        # 初始化（可选）
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, h):
        """
        参数
        -------
        h : Tensor, shape = [N, hid_dim]
            输入节点嵌入，通常来自 Encoder 或 Prompt

        返回
        -------
        logits : Tensor, shape = [N, num_classes]
            未归一化输出，可直接送入 nn.CrossEntropyLoss
        emb : Tensor, shape = [N, inner_dim]
            倒数第二层特征，可用于可视化或表征分析
        """
        emb = F.relu(self.lin1(h))
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        logits = self.lin2(emb)
        return logits, emb
