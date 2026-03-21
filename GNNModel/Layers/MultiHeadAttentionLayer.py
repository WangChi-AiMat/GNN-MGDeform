import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, node_dim, edge_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = node_dim // num_heads

        # Q, K, V
        self.W_Q = nn.Linear(node_dim, node_dim)
        self.W_K = nn.Linear(node_dim, node_dim)
        self.W_V = nn.Linear(node_dim, node_dim)

        self.fc_out = nn.Linear(node_dim, node_dim)

        self.edge_proj = nn.Linear(edge_dim, node_dim)

    def forward(self, x_i, x_j, edge_attr):
        """
        x_i, x_j: [num_edges, node_dim]
        edge_attr: [num_edges, edge_dim]
        """
        Q = self.W_Q(x_i)  # [E, node_dim]
        K = self.W_K(x_j)
        V = self.W_V(x_j)

        E = Q.size(0)
        H = self.num_heads
        Q = Q.view(E, H, self.d_k)
        K = K.view(E, H, self.d_k)
        V = V.view(E, H, self.d_k)

        edge_bias = self.edge_proj(edge_attr).view(E, H, self.d_k)

        # (Q·K + edge)
        attn_scores = (Q * (K + edge_bias)).sum(dim=-1) / (self.d_k ** 0.5)  # [E, H]
        attn_weights = F.softmax(attn_scores, dim=0)

        out = attn_weights.unsqueeze(-1) * V  # [E, H, d_k]
        out = out.view(E, -1)
        out = self.fc_out(out)
        return out