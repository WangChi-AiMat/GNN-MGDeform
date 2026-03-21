import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from .MultiHeadAttentionLayer import MultiHeadAttentionLayer

class GNNLayer(MessagePassing):
    def __init__(self, node_emb_dim, edge_emb_dim, hidden_dim, num_heads=4):
        super().__init__(aggr="add")
        from .EdgeUpdateLayer import EdgeUpdate
        from .NodeUpdate import NodeUpdate

        self.edge_update_module = EdgeUpdate(edge_emb_dim, node_emb_dim, hidden_dim)
        self.node_update_module = NodeUpdate(node_emb_dim, edge_emb_dim, hidden_dim)

        self.attention = MultiHeadAttentionLayer(node_emb_dim, edge_emb_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge_index, edge_attr, init_x_enc, init_edge_enc):
        updated_x = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr,
            init_x_enc=init_x_enc,
            init_edge_enc=init_edge_enc,
        )
        updated_x = self.dropout(updated_x)

        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        updated_edge = self.message(x_j=dst, x_i=src, edge_attr=edge_attr, init_edge_enc=init_edge_enc)
        return updated_x, updated_edge

    def message(self, x_i, x_j, edge_attr, init_edge_enc):
        attn_msg = self.attention(x_i, x_j, edge_attr)
        edge_msg = self.edge_update_module(edge_attr, x_i, x_j, init_edge_enc)
        return attn_msg + edge_msg

    def update(self, aggr_out, x, init_x_enc):
        return self.node_update_module(x, aggr_out, init_x_enc)