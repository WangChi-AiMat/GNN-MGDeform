import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


class EdgeUpdate(nn.Module):

    def __init__(self, edge_emb_dim: int, node_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_emb_dim + node_emb_dim + node_emb_dim + edge_emb_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1) ,
            nn.Linear(hidden_dim, edge_emb_dim)
        )

    def forward(self, edge_attr, src, dst,
                init_edge_enc):

        edge_input = torch.cat([
            edge_attr,
            src,
            dst,
            init_edge_enc
        ], dim=1)
        return self.mlp(edge_input)