import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F



class GraphEncoder_Attn(nn.Module):
    def __init__(self, k, d_graph, seq_len, d_data, graph_data, node_count, device, heads=3, dropout=0.1):
        super(GraphEncoder_Attn, self).__init__()
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
        self.dropout = nn.Dropout(dropout)
        self.edge_index = graph_data.edge_index.to(device)

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
        out = self.dropout(out)
        out = self.project(out.permute(0, 2, 1)).permute(0, 2, 1)    # [B, N, seq_len]
        out = self.temp_project(out)                                 # [B, N, d_graph]
        return out


class GraphEncoder_Attn_new(nn.Module):
    def __init__(
        self,
        k: int,
        d_graph: int,
        seq_len: int,
        d_data: int,
        device: torch.device,
        node_count: int,
        node_feats: torch.Tensor,
        graph_data,
        num_layers: int,
        node_feat_size=8,
        heads: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.k = k
        self.seq_len = seq_len
        self.d_graph = d_graph
        self.d_data = d_data
        self.dropout = dropout
        self.device = device
        self.node_feats = node_feats
        self.node_feat_size = node_feat_size
        # GAT edge index
        self.edge_index = graph_data.edge_index

        # --- feature-level attention to go from A×K -> d_data ---
        self.key_linear = nn.Linear((k - 1) * node_feat_size, d_data)
        self.value_linear = nn.Linear((k - 1) * node_feat_size, d_data)
        self.feature_query = nn.Parameter(torch.randn(d_data))

        # --- GAT layers ---
        self.convs = nn.ModuleList()
        # input
        self.convs.append(GATConv(d_data, d_data, heads=heads))
        # hidden
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(d_data * heads, d_data, heads=heads))
        # output
        self.convs.append(
            GATConv(d_data * heads, d_graph, heads=heads, concat=False)
        )

        # --- time-step attention to pool N -> seq_len × d_graph ---
        self.time_queries = nn.Parameter(torch.randn(seq_len, d_graph))
        self.attn_score = nn.Linear(d_graph, d_graph)

    def forward(self, mask):
        """
        features: list of length B; each is a list of N tensors [A_i, K]
        mask: (B, N) binary
        returns: (B, seq_len, d_graph)
        """
        B, N = mask.shape
        _, D_max, k_1 = self.node_feats.shape
        out = []
        for b in range(B):
            idx = torch.randint(0, D_max, (N, self.node_feat_size), device=self.device)
            idx = idx.unsqueeze(-1).expand(-1, -1, k_1)
            x = torch.gather(self.node_feats, dim=1, index=idx)  # [N, F, K_1]

            x = x.view(N, -1)  # flatten to [N, F*K_1]
            x = self.value_linear(x)  # project to [N, d_data]

            for conv in self.convs[:-1]:
                x = F.dropout(x, self.dropout, self.training)
                x = conv(x, self.edge_index)
                x = F.elu(x)

            x = F.dropout(x, self.dropout, self.training)
            x = self.convs[-1](x, self.edge_index)   # [N, d_graph]
            x *= mask[b].unsqueeze(-1)
            # 3) time-query attention pooling
            Q = self.time_queries                     # [seq_len, d_graph]
            K = self.attn_score(x)                    # [N, d_graph]
            scores = (Q @ K.T) / (self.d_graph**0.5)  # [seq_len, N]
            w = F.softmax(scores, dim=-1)             # [seq_len, N]
            gp = w @ x                                # [seq_len, d_graph]

            out.append(gp)

        return torch.stack(out, dim=0)                # [B, seq_len, d_graph]

