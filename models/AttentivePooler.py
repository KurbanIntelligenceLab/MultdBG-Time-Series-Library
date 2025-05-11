import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.data_enc = DataEmbedding(c_in=configs.enc_in,
                                           d_model=configs.d_model,
                                           embed_type=configs.embed,
                                           freq=configs.freq,
                                           dropout=configs.dropout)
        self.dBG_enc = configs.graph_encoder
        hidden_dim = configs.d_graph
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.project = nn.Linear(hidden_dim, (self.seq_len + self.pred_len) * self.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dbg_mask):
        # x_enc = (B, T, C)
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        graph_encs = []
        for i, dBG_enc in enumerate(self.dBG_enc):
            graph_encs.append(dBG_enc(dbg_mask[i]))

        dbg_out = torch.stack(graph_encs, dim=0).mean(dim=0)

        # Attentive pooling
        attn_weights = F.softmax(self.attention(dbg_out), dim=1) # (B, T, 1)
        pooled = torch.sum(attn_weights * dbg_out, dim=1)        # (B, D+G)

        # Project to forecast
        dec_out = self.project(pooled)                         # (B, P * C)
        dec_out = dec_out.view(-1, (self.pred_len + self.seq_len), self.c_out)  # (B, P, C)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raise NotImplementedError

    def anomaly_detection(self, x_enc):
        raise NotImplementedError

    def classification(self, x_enc, x_mark_enc):
        raise NotImplementedError

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dbg_mask, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, dbg_mask)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
