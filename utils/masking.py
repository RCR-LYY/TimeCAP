import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1) # 创建一个全是True的矩阵，并从主对角线的上一个对角线开始取上三角矩阵返回，返回值上三角矩阵是True，下三角矩阵是False
    # in nn.MultiheadAttention
    # binary mask is used and True means not allowed to attend
    # so we use triu instead of tril
    return mask


def generate_self_only_mask(seq_len):
    # Initialize a matrix with all True values (no attention allowed)
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    # Set the diagonal to False (allow self-attention)
    mask.fill_diagonal_(False)
    return mask


def generate_partial_mask(seq_len, mask_ratio):
    """Generate a partial mask with given mask ratio for the lower triangular part.

    Args:
        seq_len: Length of the sequence
        mask_ratio: Float between 0 and 1, portion of lower triangular elements to mask
                   0 -> causal mask (no masking in lower triangular)
                   1 -> self-only mask (all lower triangular masked)

    Returns:
        torch.Tensor: Boolean mask where True indicates positions to be masked
    """
    # Start with causal mask (upper triangular is True)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    # Get lower triangular indices (excluding diagonal)
    lower_indices = torch.tril_indices(seq_len, seq_len, offset=-1)

    # Randomly select positions to mask in lower triangular part
    num_lower = lower_indices.shape[1]
    num_to_mask = int(num_lower * mask_ratio)
    mask_indices = torch.randperm(num_lower)[:num_to_mask]

    # Set selected lower triangular positions to True (masked)
    mask[lower_indices[0][mask_indices], lower_indices[1][mask_indices]] = True

    return mask


# class TriangularCausalMask():
#     def __init__(self, B, L, device="cpu"):
#         mask_shape = [B, 1, L, L]
#         with torch.no_grad():
#             self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device) # 上三角为True，不含主对角线 triangle upper, 表示时间步之间只能看之前的时间步
#
#     @property
#     def mask(self):
#         return self._mask


# class TimerMultivariateMask():  # 最终为False的元素表示允许关注的
#     def __init__(self, B, n_vars, group_num, n_tokens, device="cpu"):
#         mask_shape = [B, 1, n_tokens, n_tokens]  # b, 1, p, p
#         with torch.no_grad():
#             self._mask1 = torch.ones((n_vars, group_num), dtype=torch.bool).to(device)  # 构建一个c, c的全True矩阵， 目的是控制变量之间如何关注，表示通道能关注任意其他通道
#             self._mask2 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)  # 上三角为True，不含主对角线 triangle upper, 表示时间步之间只能看之前的时间步
#             self._mask = torch.kron(self._mask1, self._mask2) # b, 1, n_vars*p, group_num*p
#
#     @property
#     def mask(self):
#         return self._mask


class TimerCovariateMask():  # 分组因果掩码, 允许自关注及关注相同通道的所有之前的时间块，协变量允许关注全部
    def __init__(self, B, n_vars, n_tokens, device="cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]  # b, 1, p, p
        with torch.no_grad():
            self._mask1 = torch.eye(n_vars, dtype=torch.bool).to(
                device)  # 构建一个c, c的单位矩阵，主对角线为True，其余为False, 目的是控制变量之间如何关注，
            self._mask2 = torch.tril(torch.ones(mask_shape, dtype=torch.bool)).to(
                device)  # 下三角为True，含主对角线 triangular lower
            self._mask = ~torch.kron(self._mask1, self._mask2)  # Kronecker积 克罗内克积，取反操作
            self._mask[:, :, -n_tokens:, :-n_tokens] = False  #

    @property
    def mask(self):
        return self._mask


class MultivariateMaskSelf:
    def __init__(self, n_vars, n_tokens, scope, device="cpu"):
        with torch.no_grad():
            self._mask1 = ~torch.eye(n_vars, dtype=torch.bool, device=device)
            self._mask2 = torch.tril(torch.ones((n_tokens, n_tokens), dtype=torch.bool, device=device), diagonal=0) & torch.triu(torch.ones((n_tokens, n_tokens), dtype=torch.bool, device=device), diagonal=scope)
            self._mask = torch.kron(self._mask1, self._mask2)
            self._mask.fill_diagonal_(True)
            self._mask = ~self._mask.unsqueeze(0).unsqueeze(1)

    @property
    def mask(self):
        return self._mask  # True 表示不关注该位置


# 交叉注意力肯定是不同通道的token之间计算注意力
class MultivariateMaskCross():
    def __init__(self, n_vars, group_num, n_tokens, scope, device="cpu"):
        mask_shape = [1, 1, n_tokens, n_tokens]  # b, 1, p, p
        with torch.no_grad():
            self._mask1 = torch.ones((n_vars, group_num), dtype=torch.bool, device=device)
            self._mask2 = torch.tril(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=0) & torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=scope) # +x表示主对角线以上x，-x表示主对角线以下x
            self._mask = ~torch.kron(self._mask1, self._mask2)

    @property
    def mask(self):
        return self._mask # True = 不关注