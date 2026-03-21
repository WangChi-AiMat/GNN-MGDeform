import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class NodeUpdate(nn.Module):

    def __init__(self, node_emb_dim: int, edge_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_emb_dim + edge_emb_dim + node_emb_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1) ,
            nn.Linear(hidden_dim, node_emb_dim)
        )

    def forward(self, x: torch.Tensor, aggr_out: torch.Tensor, init_x_enc: torch.Tensor) -> torch.Tensor:

        node_input = torch.cat([x, aggr_out, init_x_enc], dim=1)
        return self.mlp(node_input)