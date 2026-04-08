import torch
import math
import numpy as np
from math import sqrt
import torch.nn as nn
from einops import repeat
import torch.nn.functional as F
from layers.Attn_Bias import ChannelAttentionBias
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from utils.masking import MultivariateMaskCross


class DataEmbedding_MultiScale(nn.Module):
    def __init__(self, window_size, stride_channel, patch_len, stride_time, patch_num, d_model, dropout=0.1):
        super(DataEmbedding_MultiScale, self).__init__()
        self.window_size = window_size
        self.stride_channel = stride_channel

        self.patch_len = patch_len
        self.stride_time = stride_time
        self.patch_num = patch_num

        self.d_model = d_model
        self.value_embedding = nn.Linear(patch_len, d_model)

        self.group_channel = nn.Parameter(torch.randn(1, 1, patch_num, d_model))  # 1, 1, p, d
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x): # b, l, c
        # Channel-wise Group Independence
        x = x.unfold(dimension=-1, size=self.window_size, step=self.stride_channel)  # b, l, k, w
        x = x.permute(0, 2, 3, 1)  # b, k, w, l
        B, K, _, _ = x.shape
        x = x.flatten(0, 1)  # bk, w, l

        # Time-wise Patch Embedding
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride_time)  # bk, w, p, pl
        x = self.value_embedding(x)  # bk, w, p, d

        # Introduce a Learnable Group Channel
        group_channel = self.group_channel.repeat(B * K, 1, 1, 1)  # bk, 1, p, d
        x = torch.cat([x, group_channel], dim=1)  # bk, w+1, p, d

        # flatten
        x = x.flatten(1, 2) # bk, (w+1)p, d

        return self.dropout(x)


class ChannelBlock(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(ChannelBlock, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, n_vars, n_tokens, group_num, attn_mask=None): # bk, (w+1)*p, d
        attns = []

        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, n_vars, n_tokens, group_num, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class ChannelLayer(nn.Module):
    def __init__(self, attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(ChannelLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = cross_attention
        self.norm2 = nn.LayerNorm(d_model)

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, n_vars, n_tokens, group_num, attn_mask=None): # bk, (w+1)p, d
        attns = []

        # Intra-Group Multivariate Attention
        new_x, attn = self.attention(
            x, x, x,
            n_vars=n_vars,
            group_num=n_vars,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
            state='self-attention',
        )
        attns.append(attn)
        x = self.norm1(x + self.dropout(new_x)) # bk, (w+1)p, d

        # Channel Splitting & Context Reshaping
        x_data = x[:, :(n_vars-1) * n_tokens, :]    # bk, wp, d
        x_context = x[:, (n_vars-1) * n_tokens:, :] # bk, p, d
        x_context = x_context.contiguous().view(-1, group_num, n_tokens, x_context.shape[-1])  # b, k, p, d
        x_context = x_context.flatten(1, 2)  # b, kp, d
        x_context = x_context.repeat_interleave(group_num, dim=0)  # bk, kp, d

        # Inter-Group Multivariate Attention
        new_x_data, attn_x_data = self.cross_attention(
            x_data, x_context, x_context,
            n_vars=n_vars-1,
            group_num=group_num,
            n_tokens=n_tokens,
            attn_mask=attn_mask,
            state='cross-attention',
        )
        attns.append(attn_x_data)
        x_data = x_data + self.dropout(new_x_data)  # bk, wp, d

        # Concatenate Back & norm layer
        x_context = x_context[::group_num, :, :] # b, kp, d
        x_context = x_context.contiguous().view(-1, group_num, n_tokens, x_context.shape[-1])  # b, k, p, d
        x_context = x_context.flatten(0, 1)  # bk, p, d
        x = torch.cat([x_data, x_context], dim=1) # bk, (w+1)p, d
        x = self.norm2(x)

        # FFN
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y), attns


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.inner_attention = attention

        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None, n_vars=None, group_num=None, n_tokens=None, state=None):
        B, L, _ = queries.shape # bk, n_vars*n_tokens, d
        _, S, _ = keys.shape # bk, group_num*n_tokens, d
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1) # bk, n_vars*n_tokens, H, dq
        keys = self.key_projection(keys).view(B, S, H, -1) # bk, group_num*n_tokens, H, dk
        values = self.value_projection(values).view(B, S, H, -1) # bk, group_num*n_tokens, H, dv

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            group_num=group_num,
            n_tokens=n_tokens,
            state=state,
        ) # bk, n_vars*n_tokens, H, dq
        out = out.view(B, L, -1) # bk, n_vars*n_tokens, H*dv

        return self.out_projection(out), attn # bk, n_vars*n_tokens, d


class ChannelAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, scope=None, max_len=100, covariate=False, flash_attention=False):
        super(ChannelAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len), partial_factor=(0.0, 0.5))
        self.channel_attn_bias = ChannelAttentionBias(dim=d_model, num_heads=num_heads)
        self.scope = scope

    def forward(self, queries, keys, values, attn_mask, n_vars, group_num, n_tokens, state=None):
        B, L, H, E = queries.shape # bk, n_vars*n_tokens, h, dq
        _, S, _, D = values.shape # bk, group_num*n_tokens, h, dv

        queries = queries.permute(0, 2, 1, 3) # bk, h, n_vars*n_tokens, dq
        keys = keys.permute(0, 2, 1, 3) # bk, h, group_num*n_tokens, dk

        if state == 'self-attention':
            q_seq_id = torch.arange(n_tokens * n_vars) # [0, 1, 2, ......, L - 1]
            q_seq_id = repeat(q_seq_id, 'n -> b h n', b=B, h=H) # bk, h, l
            kv_seq_id = torch.arange(n_tokens * group_num) # [0, 1, 2, ......, s - 1]
            kv_seq_id = repeat(kv_seq_id, 'n -> b h n', b=B, h=H) # bk, h, s
            queries, keys = self.qk_proj(queries, keys, query_id=q_seq_id, kv_id=kv_seq_id)

        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("bhle,bhse->bhls", queries, keys)

        # Compute Channel Attention mask
        attn_mask = MultivariateMaskCross(n_vars, group_num, n_tokens, scope=self.scope, device=queries.device)
        scores.masked_fill_(attn_mask.mask, -np.inf)
        if self.scope < 0:
            attn_bias = self.channel_attn_bias(B=B, n_vars=n_vars, group_num=group_num, n_tokens=n_tokens, device=queries.device)
            scores += attn_bias

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class Decoder_TimeCAP(nn.Module):
    def __init__(self, d_model, patch_len, patch_num, stride_channel, OS_pred_lenth, has_os_head, dropout=0.1):
        super(Decoder_TimeCAP, self).__init__()
        self.patch_num = patch_num
        self.stride_channel = stride_channel
        self.ARG_Head = nn.Linear(d_model, patch_len)

        self.has_os_head = has_os_head
        if has_os_head:
            self.OSG_Head = nn.Linear(patch_num * d_model, OS_pred_lenth)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, group_num, enc_in, activate_os_head=False):
        # Remove global token
        x = x[:, :-self.patch_num, :]

        # Reshape to group windowed patches
        b = x.size(0) // group_num
        wp, d = x.size(1), x.size(2)
        x = x.view(b, group_num, wp, d)

        # 构造索引
        base_indices = torch.arange(0, group_num * self.stride_channel * self.patch_num, self.stride_channel * self.patch_num, device=x.device)
        offset_indices = torch.arange(wp, device=x.device)
        scatter_idx = (base_indices.view(1, -1, 1) + offset_indices.view(1, 1, -1)).expand(b, -1, -1)
        flat_idx = scatter_idx.reshape(b, -1)

        flat_x = x.reshape(b, -1, d)

        recon = torch.zeros(b, enc_in * self.patch_num, d, device=x.device)
        count = torch.zeros(b, enc_in * self.patch_num, d, device=x.device)
        recon.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, d), flat_x)
        count.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, d), torch.ones_like(flat_x))
        x = recon / count.clamp(min=1e-8)
        x = x.view(b, enc_in, self.patch_num, d)

        # ARG Head
        x_AR = self.ARG_Head(x)
        x_AR = x_AR.view(b, enc_in, -1)

        # OSG Head
        x_OS = None
        if self.has_os_head and activate_os_head:
            x_OS = self.OSG_Head(x.flatten(-2, -1))
            x_OS = self.dropout(x_OS.permute(0, 2, 1))

        return self.dropout(x_AR), x_OS
