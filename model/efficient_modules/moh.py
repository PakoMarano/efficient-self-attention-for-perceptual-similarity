import torch
import torch.nn as nn
import torch.nn.functional as F


class MoHAttention(nn.Module):
    """
    Mixture of Heads (MoH) Attention module.
    Dynamically selects top-k most important heads based on query scores,
    computing attention only for selected heads to improve efficiency.
    """
    
    def __init__(self, original_attention: nn.Module, topk_heads: int = 4):
        """
        Args:
            original_attention: The original Attention module to copy parameters from
            topk_heads: Number of heads to keep (default: 4 out of 12)
        """
        super().__init__()
        if topk_heads <= 0:
            raise ValueError("topk_heads must be > 0")
        self.num_heads = original_attention.num_heads
        self.scale = original_attention.scale
        self.topk_heads = min(topk_heads, self.num_heads)
        
        # Copy QKV projection and output projection from original
        self.qkv = original_attention.qkv
        self.attn_drop = original_attention.attn_drop
        self.proj = original_attention.proj
        self.proj_drop = original_attention.proj_drop
        
    def select_top_heads(self, q):
        """
        Select top-k most important heads based on query magnitudes.
        
        NOTE: This is a HEURISTIC routing mechanism that simplifies the full MoH approach.
        The original MoH paper uses trainable gating networks, but here we use a 
        fixed scoring function based on query energy for simplicity and zero-shot application
        to pretrained models without additional training.
        
        Args:
            q: (B, H, N, D) query tensor
        Returns:
            indices: (topk,) tensor of selected head indices
        """
        B, H, N, D = q.shape
        
        # Compute importance score per head (based on L2 norm of queries)
        # Average across batch and tokens
        # RATIONALE: Higher query energy (||Q||²) suggests the head is trying to 
        # express stronger semantic information. In attention, Q represents "what I'm 
        # looking for" - larger magnitude queries indicate the head is actively searching
        # for specific patterns, while low-energy queries suggest the head may be less
        # discriminative or redundant. This heuristic correlates with head importance
        # in information retrieval terms.
        scores = q.float().pow(2).mean(dim=(0, 2, 3))  # (H,)
        
        # Select top-k heads
        topk = min(self.topk_heads, H)
        idx = torch.topk(scores, k=topk, largest=True).indices  # (topk,)
        
        # Sort indices for cache-friendly access
        idx, _ = torch.sort(idx)
        
        return idx
    
    def forward(self, x):
        """
        Forward pass with dynamic head selection.
        
        Args:
            x: (B, N, C) where N = num_patches + 1 (includes CLS token)
        Returns:
            output: (B, N, C)
            attn: attention weights (sparse, only for selected heads)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D)
        
        # Select top-k heads based on query importance
        head_indices = self.select_top_heads(q)  # (topk,)
        
        # Extract selected heads
        q_selected = q[:, head_indices, :, :]  # (B, topk, N, D)
        k_selected = k[:, head_indices, :, :]
        v_selected = v[:, head_indices, :, :]
        
        # Compute attention only for selected heads
        attn_selected = (q_selected @ k_selected.transpose(-2, -1)) * self.scale
        attn_selected = attn_selected.softmax(dim=-1)
        attn_selected = self.attn_drop(attn_selected)
        
        # Apply attention to values
        y_selected = (attn_selected @ v_selected)  # (B, topk, N, D)
        
        # Create full output with zeros for pruned heads
        y_full = torch.zeros(B, self.num_heads, N, C // self.num_heads, 
                            device=x.device, dtype=x.dtype)
        y_full[:, head_indices, :, :] = y_selected
        
        # Reshape and project
        x_out = y_full.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        # Create sparse attention tensor (with zeros for pruned heads)
        attn_full = torch.zeros(B, self.num_heads, N, N, 
                               device=x.device, dtype=x.dtype)
        attn_full[:, head_indices, :, :] = attn_selected
        
        return x_out, attn_full


def build_attention(original_attention: nn.Module, topk_heads: int = 4, **_kwargs) -> nn.Module:
    """
    Builder function for MoH attention module.
    
    Args:
        original_attention: The standard Attention module to replace
        topk_heads: Number of heads to keep active (default: 4 out of 12)
    Returns:
        MoHAttention module with copied parameters
    """
    return MoHAttention(original_attention, topk_heads=topk_heads)