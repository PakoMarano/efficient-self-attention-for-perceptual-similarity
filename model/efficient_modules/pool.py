import torch
from torch import nn


class PoolAttention(nn.Module):
    """
    Lightweight pooling-based token mixer for ViT blocks.

    This module does not compute Q/K/V. It applies local average pooling to
    patch tokens and uses global average pooled patch content for the CLS token.
    The surrounding Transformer block applies the residual connection.
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

    def _spatial_pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        S = int(N ** 0.5)

        if S * S != N:
            raise ValueError(
                f"PoolAttention expects square patch grid, got {N} patch tokens."
            )

        patch_tokens_2d = patch_tokens.reshape(B, S, S, C).permute(0, 3, 1, 2).contiguous()
        pooled_2d = self.pool(patch_tokens_2d)
        pooled_patches = pooled_2d.flatten(2).transpose(1, 2)
        return pooled_patches

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
        - patch tokens -> local spatial average pooling
        - CLS token -> global average over patch tokens
        Returns token updates; residual addition is handled by the caller block.
        """
        cls_token, patch_tokens = x[:, :1], x[:, 1:]

        # Spatial pooling for patches
        pooled_patches = self._spatial_pool_patch_tokens(patch_tokens)
        
        # Global average pooling for CLS (adapts MetaFormer to ViT structure)
        # CLS gets updated with global mean: cls_out = cls_in + mean(norm(patches))
        global_avg_patches = patch_tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        token_updates = torch.cat([global_avg_patches, pooled_patches], dim=1)

        return token_updates, None


def build_attention(original_attention: nn.Module, kernel_size: int = 3, **_kwargs) -> nn.Module:
    return PoolAttention(kernel_size=kernel_size)