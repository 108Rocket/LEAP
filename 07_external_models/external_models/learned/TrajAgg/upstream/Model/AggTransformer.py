import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import mean_pooling, creat_padding_mask



class AggAttn(nn.Module):
    def __init__(self, d_model, num_heads):
        super(AggAttn, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)


    def forward(self, q, k, v, attention_mask=None, src_key_padding_mask=None):
        B, L_q, _ = q.size()
        _, L_k, _ = k.size()

        # Linear projection
        q = self.q_proj(q)  # [B, L_q, D]
        k = self.k_proj(k)  # [B, L_k, D]
        v = self.v_proj(v)  # [B, L_k, D]

        # Split into heads
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D_h]
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D_h]
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D_h]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L_q, L_k]

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores += attention_mask

        if src_key_padding_mask is not None:
            if src_key_padding_mask.dim() == 2:
                mask = (src_key_padding_mask.unsqueeze(1) & src_key_padding_mask.unsqueeze(2)).unsqueeze(1)
            elif src_key_padding_mask.dim() == 3:
                mask = src_key_padding_mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax
        attn_weights = torch.nan_to_num(scores, nan=0.0)  # 替换 NaN 为 0
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, H, L_q, L_k]
        attn_output = attn_weights @ v  # [B, H, L_q, D_h]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)  # [B, L_q, D]
        return attn_output, attn_weights



class AggAttnLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AggAttnLayer, self).__init__()
        self.aggattn = AggAttn(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, 2 * d_model)
        self.linear2 = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, q, kv, attention_mask=None, src_key_padding_mask=None):
        q_, kv_ = q, kv
        q, _ = self.aggattn(q, kv_, kv_, attention_mask, src_key_padding_mask)
        kv, _ = self.aggattn(kv, q_, q_, attention_mask, src_key_padding_mask)

        # Feed-forward block
        q1 = self.linear2(self.dropout(F.relu(self.linear1(q))))
        kv1 = self.linear2(self.dropout(F.relu(self.linear1(kv))))
        q = self.norm1(q + self.dropout(q1))
        kv = self.norm1(kv + self.dropout(kv1))
        return q, kv


class AggAttnEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout, mu, metric):
        super(AggAttnEncoder, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(2, d_model)
        self.encoder_seq = nn.LSTM(2, d_model, num_layers, batch_first=True)
        self.aggattnlayer = nn.ModuleList([AggAttnLayer(d_model, num_heads, dropout) for _ in range(num_layers)])

        _ = nn.TransformerEncoderLayer(d_model, nhead=num_heads, batch_first=True)
        self.T = nn.TransformerEncoder(_, num_layers)

        self.mu = mu
        self.metric = metric

    def forward(self, q, kv, attention_mask=None, src_key_padding_mask=None):

        q, kv = self.encoder_seq(q)[0], self.encoder_seq(kv)[0]

        for layer in self.aggattnlayer:
            q, kv = layer(q, kv, attention_mask, src_key_padding_mask)

        q, kv = mean_pooling(q, src_key_padding_mask), mean_pooling(kv, src_key_padding_mask)
        out = self.mu * q + (1 - self.mu) * kv
        return out

