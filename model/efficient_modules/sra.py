import torch
import torch.nn as nn
import torch.nn.functional as F


class SRAAttention(nn.Module):
    """
    Spatial Reduction Attention (SRA) module.
    Reduces spatial dimensions of K and V using average pooling to decrease computational cost.
    """
    
    def __init__(self, original_attention: nn.Module, reduction_ratio: int = 4):
        """
        Args:
            original_attention: The original Attention module to copy parameters from
            reduction_ratio: Spatial reduction factor for K and V (default: 4)
        """
        super().__init__()
        self.num_heads = original_attention.num_heads
        self.scale = original_attention.scale
        self.reduction_ratio = reduction_ratio
        
        # Copy QKV projection and output projection from original
        self.qkv = original_attention.qkv
        self.attn_drop = original_attention.attn_drop
        self.proj = original_attention.proj
        self.proj_drop = original_attention.proj_drop
        
    def spatial_reduction(self, k, v):
        """
        Apply spatial reduction to K and V using average pooling.
        
        Args:
            k, v: (B, H, N, D) where N = S*S (patch tokens only, no CLS)
        Returns:
            Reduced k, v: (B, H, Nr, D) where Nr = (S//r)*(S//r)
        """
        B, H, N, D = k.shape
        S = int(N ** 0.5)
        
        if S * S != N:
            # Cannot reshape to square grid, return original
            return k, v
        
        # Reshape to spatial grid: (B, H, S, S, D)
        k_spatial = k.view(B, H, S, S, D)
        v_spatial = v.view(B, H, S, S, D)
        
        # Permute to (B, H, D, S, S) for spatial pooling
        k_spatial = k_spatial.permute(0, 1, 4, 2, 3).contiguous()
        v_spatial = v_spatial.permute(0, 1, 4, 2, 3).contiguous()
        
        # Flatten B and H for pooling
        k_flat = k_spatial.flatten(0, 1)  # (B*H, D, S, S)
        v_flat = v_spatial.flatten(0, 1)
        
        # Apply average pooling
        k_pooled = F.avg_pool2d(
            k_flat,
            kernel_size=self.reduction_ratio,
            stride=self.reduction_ratio
        )
        v_pooled = F.avg_pool2d(
            v_flat,
            kernel_size=self.reduction_ratio,
            stride=self.reduction_ratio
        )
        
        # Reshape back: (B, H, D, S//r, S//r)
        S_reduced = S // self.reduction_ratio
        k_pooled = k_pooled.view(B, H, D, S_reduced, S_reduced)
        v_pooled = v_pooled.view(B, H, D, S_reduced, S_reduced)
        
        # Permute back to (B, H, S//r, S//r, D) and flatten spatial
        k_reduced = k_pooled.permute(0, 1, 3, 4, 2).contiguous()
        v_reduced = v_pooled.permute(0, 1, 3, 4, 2).contiguous()
        
        # Flatten to (B, H, Nr, D)
        k_reduced = k_reduced.flatten(2, 3)
        v_reduced = v_reduced.flatten(2, 3)
        
        return k_reduced, v_reduced
    
    def forward(self, x):
        """
        Forward pass with spatial reduction attention.
        
        Args:
            x: (B, N, C) where N = num_patches + 1 (includes CLS token)
        Returns:
            output: (B, N, C)
            attn: attention weights (for compatibility)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V from input
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D)
        
        # Split CLS token from patches
        q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]  # (B, H, 1, D), (B, H, N-1, D)
        k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
        v_cls, v_patch = v[:, :, :1, :], v[:, :, 1:, :]
        
        # Apply spatial reduction to patch tokens only
        k_reduced, v_reduced = self.spatial_reduction(k_patch, v_patch)
        
        # Combine CLS with reduced patches for K and V
        k_combined = torch.cat([k_cls, k_reduced], dim=2)  # (B, H, 1+Nr, D)
        v_combined = torch.cat([v_cls, v_reduced], dim=2)
        
        # Compute attention: all queries attend to reduced K/V
        q_all = q  # Keep all queries (CLS + patches) at full resolution
        attn = (q_all @ k_combined.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x_out = (attn @ v_combined).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out, attn


def build_attention(original_attention: nn.Module, reduction_ratio: int = 4, **_kwargs) -> nn.Module:
    """
    Builder function for SRA attention module.
    
    Args:
        original_attention: The standard Attention module to replace
        reduction_ratio: Spatial reduction factor (default: 4)
    Returns:
        SRAAttention module with copied parameters
    """
    return SRAAttention(original_attention, reduction_ratio=reduction_ratio)