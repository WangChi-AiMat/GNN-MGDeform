import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from torchmetrics import PearsonCorrCoef
from .Layers.GNNLayer import GNNLayer
import warnings
warnings.filterwarnings("ignore")

class GNNModel(nn.Module):
    """完整GNN模型：包含节点/边编码器、多层GNN和解码器"""
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim,
        node_emb_dim,
        edge_emb_dim,
        n_rec,  # 递归更新次数
        num_heads,
        decoder_hidden,
        Dropout_encoder,
        Dropout_decoder
    ):
        super().__init__()
        self.n_rec = n_rec
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim

        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.Dropout(p=Dropout_encoder),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 边编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1) ,
            nn.Dropout(p=Dropout_encoder),
            nn.Linear(hidden_dim, edge_emb_dim)
        )

        # 递归更新层
        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_emb_dim, edge_emb_dim, hidden_dim, num_heads)
            for _ in range(n_rec)
        ])

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(node_emb_dim, decoder_hidden),
            nn.LeakyReLU(negative_slope=0.1) ,
            nn.Dropout(p=Dropout_decoder),  # 全连接层后加入 Dropout
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LeakyReLU(negative_slope=0.1) ,
            nn.Dropout(p=Dropout_decoder),  # 再次加入 Dropout
            nn.Linear(decoder_hidden, 1)  # 输出预测值（如回归任务）
        )

    def forward(self, x, edge_index=None, edge_attr=None, **kwargs):
        """
        前向传播，支持两种调用方式：
        1. 训练时：model(data)  # data 是 Data 对象
        2. Explainer 时：model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        """

        # ================================
        # 1. 兼容 model(data) 调用
        # ================================
        if isinstance(x, Data):
            data = x
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr

        # ===========================================================
        # 2. 兼容 Explainer 内部调用：model(x) or model(x, edge_attr=xxx)
        # ===========================================================
        # explainer 有可能只传 x，edge_attr 放在 kwargs
        if edge_index is None:
            if "edge_index" in kwargs:
                edge_index = kwargs["edge_index"]
            else:
                raise ValueError("edge_index 不能为 None，也未在 kwargs 中找到 edge_index")

        if edge_attr is None:
            if "edge_attr" in kwargs:
                edge_attr = kwargs["edge_attr"]
            else:
                raise ValueError("edge_attr 不能为 None，也未在 kwargs 中找到 edge_attr")

        # ================================
        # 3. 编码层
        # ================================
        init_x = self.node_encoder(x)  # [num_nodes, node_emb_dim]
        init_edge = self.edge_encoder(edge_attr)  # [num_edges, edge_emb_dim]

        # ================================
        # 4. 递归 GNN 更新
        # ================================
        current_x = init_x
        current_edge = init_edge

        for layer in self.gnn_layers:
            current_x, current_edge = layer(
                x=current_x,
                edge_index=edge_index,
                edge_attr=current_edge,
                init_x_enc=init_x,
                init_edge_enc=init_edge,
            )

        # ================================
        # 5. 解码器
        # ================================
        pred = self.decoder(current_x)

        return pred
