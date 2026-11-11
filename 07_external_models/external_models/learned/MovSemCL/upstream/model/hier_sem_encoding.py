import sys
sys.path.append('..')
import math
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 201):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create learnable positional encoding parameters
        self.pos_embedding = nn.Parameter(torch.empty(maxlen, 1, emb_size))
        nn.init.xavier_uniform_(self.pos_embedding)
        # self.pos_embedding = nn.Parameter(torch.randn(maxlen, 1, emb_size))
    def forward(self, token_embedding: torch.Tensor):
        seq_len = token_embedding.size(0)
        return self.dropout(token_embedding + self.pos_embedding[:seq_len, :, :])


class HSE(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput, group_size=4):
        super(HSE, self).__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.group_size = group_size
        self.attn_dropout = attn_dropout        
        self.structural_token_pos_encoder = None
        self.spatial_token_pos_encoder = None
        
        self.structural_patch_pos_encoder = LearnablePositionalEncoding(ninput, pos_droput)
        
        self.spatial_patch_pos_encoder = LearnablePositionalEncoding(4, pos_droput)
        
        self.group_attn = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        
        spatial_nhead = min(nhead, 4)
        self.spatial_group_attn = nn.TransformerEncoderLayer(
            4, spatial_nhead, nhidden, attn_dropout
        )
        
        structural_attn_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout)
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer)
        
        # 使用新的类名
        self.spatial_attn = HierSemAttn(4, 32, 1, 3, attn_dropout, pos_droput)
        
        self.gamma_param = nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def _group_and_process(self, features, feature_type, src_padding_mask, pos_encoder=None):
        seq_len, batch_size, feature_dim = features.shape        
        num_groups = (seq_len + self.group_size - 1) // self.group_size
        
        if seq_len % self.group_size != 0:
            pad_len = num_groups * self.group_size - seq_len
            pad_features = torch.zeros(pad_len, batch_size, feature_dim, device=features.device, dtype=features.dtype)
            features_padded = torch.cat([features, pad_features], dim=0)
            pad_mask = torch.ones(batch_size, pad_len, device=src_padding_mask.device, dtype=src_padding_mask.dtype)
            src_padding_mask_padded = torch.cat([src_padding_mask, pad_mask], dim=1)
        else:
            features_padded = features
            src_padding_mask_padded = src_padding_mask
        
        features_grouped = features_padded.view(num_groups, self.group_size, batch_size, feature_dim)
        mask_grouped = src_padding_mask_padded.view(batch_size, num_groups, self.group_size)

        features_all_groups = features_grouped.transpose(0, 1).contiguous()
        features_all_groups = features_all_groups.view(self.group_size, num_groups * batch_size, feature_dim)
        
        mask_all_groups = mask_grouped.transpose(0, 1).contiguous()
        mask_all_groups = mask_all_groups.view(num_groups * batch_size, self.group_size)

        if feature_type == 'structural':
            group_output_all = self.group_attn(features_all_groups, src_key_padding_mask=mask_all_groups)
        elif feature_type == 'spatial':
            group_output_all = self.spatial_group_attn(features_all_groups, src_key_padding_mask=mask_all_groups)
        
        group_output_all = group_output_all.view(self.group_size, num_groups, batch_size, feature_dim)
        group_output_all = group_output_all.transpose(0, 1)
        
        mask_for_pool = (~mask_grouped).float()
        mask_for_pool = mask_for_pool.transpose(0, 1).unsqueeze(-1)
        mask_for_pool = mask_for_pool.transpose(1, 2)
        
        group_sum = (group_output_all * mask_for_pool).sum(dim=1)
        group_len = mask_for_pool.sum(dim=1) + 1e-8
        grouped_features = group_sum / group_len
        
        grouped_padding_mask = (mask_grouped.sum(dim=2) == self.group_size).float()
        
        return grouped_features, grouped_padding_mask

    def _group_and_aggregate(self, src, src_padding_mask):
        return self._group_and_process(src, 'structural', src_padding_mask, None)
    
    def _group_spatial_features(self, srcspatial, src_padding_mask):
        grouped_features, grouped_padding_mask = self._group_and_process(
            srcspatial, 'spatial', src_padding_mask, None
        )
        return grouped_features, grouped_padding_mask
        
    def forward(self, src, attn_mask, src_padding_mask, src_len, srcspatial):
        grouped_src, grouped_padding_mask = self._group_and_aggregate(src, src_padding_mask)

        attn_spatial = None
        if srcspatial is not None:
            grouped_spatial, _ = self._group_spatial_features(srcspatial, src_padding_mask)
            grouped_spatial_with_pos = self.spatial_patch_pos_encoder(grouped_spatial)
            
            grouped_len = torch.ceil(src_len.float() / self.group_size).long()
            
            _, attn_spatial = self.spatial_attn(grouped_spatial_with_pos, None, grouped_padding_mask, grouped_len)
            
            attn_spatial = attn_spatial.repeat(self.nhead, 1, 1)
            gamma = torch.sigmoid(self.gamma_param) * 10
            attn_spatial = gamma * attn_spatial
        
        grouped_src = self.structural_patch_pos_encoder(grouped_src)
        rtn = self.structural_attn(grouped_src, attn_spatial, grouped_padding_mask)
        
        mask = 1 - grouped_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        grouped_len = torch.ceil(src_len.float() / self.group_size) + 1e-8
        rtn = rtn / grouped_len.unsqueeze(-1).expand(rtn.shape)

        return rtn 

class HierSemAttn(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput):
        super(HierSemAttn, self).__init__()
        
        trans_encoder_layers = HierSemAttnLayer(ninput, nhead, nhidden, attn_dropout)
        self.trans_encoder = HierSemAttnEncoder(trans_encoder_layers, nlayer)
        
    def forward(self, src, attn_mask, src_padding_mask, src_len):

        rtn, attn = self.trans_encoder(src, attn_mask, src_padding_mask)

        # average pooling
        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        # rtn = [batch_size, traj_emb]
        # attn = [batch_size, seq_len, seq_len]
        return rtn, attn


class HierSemAttnEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(HierSemAttnEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


class HierSemAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(HierSemAttnLayer, self).__init__()
        self.self_attn = nn.modules.activation.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SpatialMSMLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn
