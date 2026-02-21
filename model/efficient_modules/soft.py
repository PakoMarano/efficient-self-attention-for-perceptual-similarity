import torch
import torch.nn as nn
import torch.nn.functional as F


class SOFTAttention(nn.Module):
    """
    SOFT (Scalable Orthogonal Fourier Transform) Attention module.
    Uses Nyström approximation with Gaussian kernels for linear complexity O(N·M).
    """
    
    def __init__(self, original_attention: nn.Module, reduction_ratio: int = 2, 
                 num_newton_iterations: int = 6):
        """
        Args:
            original_attention: The original Attention module to copy parameters from
            reduction_ratio: Spatial reduction factor for landmarks (default: 2)
            num_newton_iterations: Number of Newton-Raphson iterations for matrix inverse (default: 6)
        """
        super().__init__()
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be > 0")
        if num_newton_iterations <= 0:
            raise ValueError("num_newton_iterations must be > 0")
        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.qkv.in_features // original_attention.num_heads
        self.reduction_ratio = reduction_ratio
        self.num_newton_iterations = num_newton_iterations
        
        # Copy QKV projection and output projection from original
        self.qkv = original_attention.qkv
        self.attn_drop = original_attention.attn_drop
        self.proj = original_attention.proj
        self.proj_drop = original_attention.proj_drop
        
    def kernel_gauss(self, x, y, tau=None):
        """
        Computes Gaussian kernel: exp(-||x-y||² / tau)
        Using dot-product expansion: ||x-y||² = ||x||² + ||y||² - 2(x·y)
        
        Args:
            x: (B, H, N, D)
            y: (B, H, M, D)
            tau: Temperature parameter (default: sqrt(D))
        Returns:
            Kernel matrix: (B, H, N, M)
        """
        if tau is None:
            tau = self.head_dim ** 0.5
        
        # Compute squared norms
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)  # (B, H, N, 1)
        y_norm = (y ** 2).sum(dim=-1, keepdim=True)  # (B, H, M, 1)
        
        # Compute dot product
        dot = torch.matmul(x, y.transpose(-2, -1))  # (B, H, N, M)
        
        # Distance squared
        dist_sq = x_norm + y_norm.transpose(-2, -1) - 2 * dot
        
        # Apply Gaussian kernel
        return torch.exp(-dist_sq / tau)
    
    def newton_inverse(self, mat):
        """
        Computes matrix inverse using Newton-Raphson iterations.
        X_{k+1} = X_k · (2I - A · X_k)
        
        Args:
            mat: (B, H, M, M)
        Returns:
            Inverse: (B, H, M, M)
        """
        B, H, M, _ = mat.shape
        I = torch.eye(M, device=mat.device, dtype=mat.dtype).view(1, 1, M, M)
        
        # Initialize: scale · A^T
        norm = torch.max(torch.sum(torch.abs(mat), dim=-1), dim=-1, keepdim=True)[0]
        inv = mat.transpose(-2, -1) / (norm.unsqueeze(-1) * norm.unsqueeze(-2) + 1e-6)
        
        # Newton iterations
        for _ in range(self.num_newton_iterations):
            T = 2 * I - torch.matmul(mat, inv)
            inv = torch.matmul(inv, T)
        
        return inv
    
    def spatial_pool_landmarks(self, x):
        """
        Create landmarks via spatial average pooling.
        
        Args:
            x: (B, H, N, D) where N = S*S (patch tokens only)
        Returns:
            Pooled landmarks: (B, H, M, D) where M = (S//r)*(S//r)
        """
        B, H, N, D = x.shape
        S = int(N ** 0.5)
        
        if S * S != N or S < self.reduction_ratio or (S % self.reduction_ratio != 0):
            # Cannot reshape to square grid, return original
            return x
        
        # Reshape to spatial grid
        x_spatial = x.view(B, H, S, S, D)
        
        # Permute to (B, H, D, S, S) for pooling
        x_spatial = x_spatial.permute(0, 1, 4, 2, 3).contiguous()
        
        # Flatten B and H for pooling
        x_flat = x_spatial.flatten(0, 1)  # (B*H, D, S, S)
        
        # Apply average pooling
        x_pooled = F.avg_pool2d(
            x_flat,
            kernel_size=self.reduction_ratio,
            stride=self.reduction_ratio
        )
        
        # Reshape back
        S_reduced = S // self.reduction_ratio
        x_pooled = x_pooled.view(B, H, D, S_reduced, S_reduced)
        
        # Permute back and flatten spatial dims
        x_reduced = x_pooled.permute(0, 1, 3, 4, 2).contiguous()
        x_reduced = x_reduced.flatten(2, 3)  # (B, H, M, D)
        
        return x_reduced
    
    def forward(self, x):
        """
        Forward pass with SOFT attention using Nyström approximation.
        
        Args:
            x: (B, N, C) where N = num_patches + 1 (includes CLS token)
        Returns:
            output: (B, N, C)
            attn: None (SOFT doesn't compute explicit attention matrix)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, D)
        
        # Split CLS token from patches
        q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
        k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
        v_cls, v_patch = v[:, :, :1, :], v[:, :, 1:, :]
        
        # Create landmarks via pooling (Q_c, K_c)
        q_landmarks = self.spatial_pool_landmarks(q_patch)  # (B, H, M, D)
        k_landmarks = self.spatial_pool_landmarks(k_patch)  # (B, H, M, D)
        
        # Compute kernel matrices for Nyström approximation
        # M1: Kernel(Q_patch, K_landmarks) - (B, H, N-1, M)
        M1 = self.kernel_gauss(q_patch, k_landmarks)
        
        # M2: Kernel(Q_landmarks, K_landmarks) - (B, H, M, M)
        M2 = self.kernel_gauss(q_landmarks, k_landmarks)
        
        # M3: Kernel(Q_landmarks, K_patch) - (B, H, M, N-1)
        M3 = self.kernel_gauss(q_landmarks, k_patch)
        
        # Compute inverse of middle term
        M2_inv = self.newton_inverse(M2)
        
        # Linear complexity multiplication: M1 @ M2_inv @ M3 @ V
        # Order: M1 @ (M2_inv @ (M3 @ V)) for O(N·M) complexity
        
        # Step 1: Project V onto landmarks (M3 @ V_patch)
        V_projected = torch.matmul(M3, v_patch)  # (B, H, M, D)
        
        # Step 2: Mix landmarks (M2_inv @ V_projected)
        V_mixed = torch.matmul(M2_inv, V_projected)  # (B, H, M, D)
        
        # Step 3: Project back to full sequence (M1 @ V_mixed)
        y_patch = torch.matmul(M1, V_mixed)  # (B, H, N-1, D)
        
        # Handle CLS token: CLS attends to landmark-projected values
        M1_cls = self.kernel_gauss(q_cls, k_landmarks)  # (B, H, 1, M)
        y_cls = torch.matmul(M1_cls, V_mixed)  # (B, H, 1, D)
        
        # Combine CLS and patch outputs
        y = torch.cat([y_cls, y_patch], dim=2)  # (B, H, N, D)
        
        # Reshape and project
        x_out = y.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        return x_out, None  # Return None for attn (not computed explicitly)


def build_attention(original_attention: nn.Module, reduction_ratio: int = 2, 
                   num_newton_iterations: int = 6, **_kwargs) -> nn.Module:
    """
    Builder function for SOFT attention module.
    
    Args:
        original_attention: The standard Attention module to replace
        reduction_ratio: Spatial reduction factor for landmarks (default: 2)
        num_newton_iterations: Newton-Raphson iterations for inverse (default: 6)
    Returns:
        SOFTAttention module with copied parameters
    """
    return SOFTAttention(
        original_attention,
        reduction_ratio=reduction_ratio,
        num_newton_iterations=num_newton_iterations
    )