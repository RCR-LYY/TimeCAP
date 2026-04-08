import abc
import math
import torch
from einops import rearrange
from torch import nn

# 只在微调的时候用
class ChannelAttentionBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.emb = nn.Embedding(2, num_heads)

    def forward(self, B, n_vars, group_num, n_tokens, device="cpu"): # n_vars是q通道数，group_num是kv通道数， n_tokens是时间块数
        mask1 = torch.ones(n_vars, group_num, dtype=torch.bool, device=device)
        mask2 = torch.eye(n_tokens, dtype=torch.bool, device=device)
        ind = torch.kron(mask1, mask2).unsqueeze(0).unsqueeze(1)
        ind = ind.expand(B, -1, -1, -1)
        weight = rearrange(self.emb.weight, "t h -> t h 1 1")
        return (~ind) * weight[1:] + ind * weight[:1]




