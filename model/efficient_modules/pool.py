import torch
from torch import nn


class PoolAttention(nn.Module):
    """
    Lightweight MetaFormer-style token mixer.

    This module intentionally does not compute Q, K, V projections. It applies
    local average pooling over patch tokens and returns a delta token sequence
    (pool(x) - x), while keeping the CLS token unchanged.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be > 0")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        bsz, token_count, channels = patch_tokens.shape
        side = int(token_count ** 0.5)

        if side * side != token_count:
            raise ValueError(
                f"PoolAttention expects square patch grid, got {token_count} patch tokens."
            )

        patch_tokens_2d = patch_tokens.reshape(bsz, side, side, channels).permute(0, 3, 1, 2).contiguous()
        pooled_2d = self.pool(patch_tokens_2d)
        pooled_patches = pooled_2d.flatten(2).transpose(1, 2)
        return pooled_patches

    def forward(self, x: torch.Tensor):
        """
        Forward pass: spatial pool for patches, global average pool for CLS.
        This gives CLS token a path to "see" the image content.
        The Block's residual connection handles: x_out = x_in + pool(norm(x_in))
        """
        cls_token, patch_tokens = x[:, :1], x[:, 1:]

        # Spatial pooling for patches
        pooled_patches = self._pool_patch_tokens(patch_tokens)
        
        # Global average pooling for CLS (adapts MetaFormer to ViT structure)
        # CLS gets updated with global mean: cls_out = cls_in + mean(norm(patches))
        global_avg_patches = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        y = torch.cat([global_avg_patches, pooled_patches], dim=1)

        return y, None


def build_attention(original_attention: nn.Module, kernel_size: int = 3, **_kwargs) -> nn.Module:
    return PoolAttention(kernel_size=kernel_size)