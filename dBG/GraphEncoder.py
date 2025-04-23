import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphEncoder(nn.Module):
    def __init__(self, k, d_graph, seq_len, d_data, data_batch, node_count, heads=3):
        super(GraphEncoder, self).__init__()
        self.k = k
        self.seq_len = seq_len
        self.d_graph = d_graph
        self.d_data = d_data
        self.mask_proj = nn.Linear(1, self.d_data)
        self.conv1 = GATConv(
            in_channels=self.d_data,
            out_channels=self.d_data // heads,
            heads=heads,
            concat=True
        )
        self.temp_project = nn.Linear((self.d_data // heads) * heads, d_graph)
        self.project = nn.Linear(in_features=node_count, out_features=seq_len)
        self.query_proj = nn.Linear(seq_len, node_count)
        self.data_batch = data_batch


    def forward(self, x_enc, mask):
        B, N = mask.shape
        q = self.query_proj(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, DD], global query vector
        mask_embed = self.mask_proj(mask.unsqueeze(-1))  # [B, N, DD]
        self.data_batch.x = torch.cat([mask_embed[i] + q[i] for i in range(B)], dim=0)  # [N, DD]
        x = self.conv1(self.data_batch.x, self.data_batch.edge_index)
        x = x.reshape(B, N, -1)
        x = self.project(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.temp_project(x)
        return x