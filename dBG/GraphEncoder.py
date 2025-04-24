import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphEncoder(nn.Module):
    def __init__(self, k, d_graph, seq_len, d_data, graph_data, node_count, device, heads=3):
        super(GraphEncoder, self).__init__()
        self.k = k
        self.seq_len = seq_len
        self.d_graph = d_graph
        self.d_data = d_data
        self.node_count = node_count

        self.mask_proj = nn.Linear(1, self.d_data)
        self.conv1 = GATConv(in_channels=self.d_data, out_channels=self.d_data // heads, heads=heads, concat=True)
        self.temp_project = nn.Linear((self.d_data // heads) * heads, d_graph)
        self.project = nn.Linear(in_features=node_count, out_features=seq_len)
        self.query_proj = nn.Linear(seq_len, node_count)

        self.edge_index = graph_data.edge_index  # store once
        self.edge_index = self.edge_index.to(device)

    def forward(self, x_enc, mask):
        B, N = mask.shape
        q = self.query_proj(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, DD]
        mask_embed = self.mask_proj(mask.unsqueeze(-1))              # [B, N, DD]
        x = mask_embed + q                                           # [B, N, DD]

        x = x.reshape(-1, self.d_data)                               # [B*N, DD]
        edge_index = self.edge_index.repeat(1, B) + \
                     torch.arange(B, device=x.device).repeat_interleave(self.edge_index.size(1)) * N

        out = self.conv1(x, edge_index)                              # [B*N, DD]
        out = out.view(B, N, -1)                                     # [B, N, DD]
        out = self.project(out.permute(0, 2, 1)).permute(0, 2, 1)    # [B, N, seq_len]
        out = self.temp_project(out)                                 # [B, N, d_graph]
        return out
