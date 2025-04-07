import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINEConv, global_mean_pool


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class GraphEncoder(nn.Module):
    def __init__(self, k, d_graph, seq_len, global_dbg=None, num_heads=4):
        """
        Args:
            k (int): k-mer length.
            d_graph (int): Dimension used in the graph embedding.
            seq_len (int): Length of the input sequence.
            global_dbg: Global dBG graph (PyG Data object) to be fused with local subgraphs.
            num_heads (int): Number of attention heads for the fusion.
        """
        super(GraphEncoder, self).__init__()
        self.global_dbg = global_dbg  # Store global dBG for later use

        edge_feat_cnt = k + 1

        # MLP for edge features for local subgraph
        edge_mlp_local = Sequential(
            Linear(edge_feat_cnt, d_graph),
            ReLU(),
            Linear(d_graph, d_graph)
        )
        # MLP for edge features for global graph
        edge_mlp_global = Sequential(
            Linear(edge_feat_cnt, d_graph),
            ReLU(),
            Linear(d_graph, d_graph)
        )
        # Separate GINE convolution layers for local and global graphs
        self.conv_local = GINEConv(nn=edge_mlp_local, edge_dim=edge_feat_cnt)
        self.conv_global = GINEConv(nn=edge_mlp_global, edge_dim=edge_feat_cnt)

        # Multi-head attention layer for fusion:
        # The local embedding serves as the query and the global embedding as key/value.
        self.attention = nn.MultiheadAttention(embed_dim=d_graph, num_heads=num_heads, batch_first=True)
        # Projection for the attention output (used in a residual connection)
        self.attn_proj = nn.Linear(d_graph, d_graph)
        # (Optional) Additional projection layer if further adjustment of final output dimension is needed.
        self.project = nn.Linear(in_features=seq_len - k + 2, out_features=seq_len)

    def forward(self, local_data, device):
        """
        Args:
            local_data: PyG Data object for the local subgraph.
            device: The device (cpu/gpu) on which to run computations.
        Returns:
            fused_embedding: The fused graph-level embedding.
            attn_weights: The attention weights from the fusion.
        """
        # --- Process Local Subgraph ---
        x_local = torch.ones((local_data.num_nodes, 1), device=device)
        local_edge_index = local_data.edge_index.to(device)
        local_weight = local_data.weight.to(device)
        local_kmer = local_data.kmer.to(device)
        local_batch = local_data.batch.to(device)
        local_edge_attr = torch.cat([local_weight.view(-1, 1), local_kmer], dim=1).to(device).float()

        x_local = self.conv_local(x_local, local_edge_index, local_edge_attr)
        x_local = F.leaky_relu(x_local)
        local_embedding = global_mean_pool(x_local, local_batch)  # shape: (batch_size, d_graph)

        # --- Process Global Graph ---
        global_data = self.global_dbg.to(device)
        x_global = torch.ones((global_data.num_nodes, 1), device=device)
        global_edge_index = global_data.edge_index.to(device)
        global_weight = global_data.weight.to(device)
        global_kmer = global_data.kmer.to(device)
        # Create a dummy batch vector for the global graph: all nodes belong to the same graph (index 0)
        global_batch = torch.zeros(global_data.num_nodes, dtype=torch.long, device=device)
        global_edge_attr = torch.cat([global_weight.view(-1, 1), global_kmer], dim=1).to(device).float()

        x_global = self.conv_global(x_global, global_edge_index, global_edge_attr)
        x_global = F.leaky_relu(x_global)
        global_embedding = global_mean_pool(x_global, global_batch)  # shape: (1, d_graph)

        # --- Fuse Local and Global Representations with Multi-Head Attention ---
        # Expand global embedding so its shape matches the local embedding batch size.
        # Since the global graph is shared, we repeat it for each local sample.
        batch_size = local_embedding.shape[0]
        global_embedding_expanded = global_embedding.repeat(batch_size, 1)  # shape: (batch_size, d_graph)

        query = local_embedding.unsqueeze(1)  # shape: (batch_size, 1, d_graph)
        key = global_embedding_expanded.unsqueeze(1)  # shape: (batch_size, 1, d_graph)
        value = global_embedding_expanded.unsqueeze(1)  # shape: (batch_size, 1, d_graph)

        attn_output, attn_weights = self.attention(query, key, value)
        fused_embedding = query + self.attn_proj(attn_output)
        fused_embedding = fused_embedding.permute(1, 2, 0)
        fused_embedding = self.project(fused_embedding).permute(0, 2, 1)
        return fused_embedding, attn_weights


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
