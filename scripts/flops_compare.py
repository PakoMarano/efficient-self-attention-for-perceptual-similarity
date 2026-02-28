import argparse
from dataclasses import dataclass

"""
Theoretical full-model MACs/FLOPs comparison for MHA vs Pool token mixer.

What this script prints:
- full-model approximate MACs/FLOPs for a ViT backbone:
    patch embedding + transformer blocks (token mixer + MLP), for both `mha` and `pool`.

Assumptions:
- square image and square patch grid (`img_size % patch_size == 0`);
- MHA block MACs: 4 * N * d^2 + 2 * N^2 * d;
- Pool block MACs follow `model/efficient_modules/pool.py`:
    patch local average pooling + CLS global average over patch tokens (kernel size = 3);
- MLP MACs per block: 2 * N * d * (mlp_ratio * d);
- RGB patch embedding input channels fixed to 3;
- 1 MAC = 2 FLOPs convention;
- lower-order ops are ignored in full-model approximation
    (e.g., LayerNorm, bias adds, residual adds).
"""

POOL_KERNEL_SIZE = 3
IN_CHANNELS = 3


@dataclass
class FlopsStats:
    tokens_total: int
    tokens_patch: int
    patch_embed_macs: int
    mlp_macs_per_block: int
    mha_model_macs_total: int
    pool_model_macs_total: int
    mha_model_flops_total: int
    pool_model_flops_total: int
    model_mac_ratio_mha_over_pool: float
    model_savings_fraction: float


def _num_tokens(img_size: int, patch_size: int) -> tuple[int, int]:
    if img_size <= 0:
        raise ValueError("img_size must be > 0")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if img_size % patch_size != 0:
        raise ValueError("img_size must be divisible by patch_size for square-grid ViT tokens")

    side_tokens = img_size // patch_size
    patch_tokens = side_tokens * side_tokens
    total_tokens = patch_tokens + 1  # + CLS
    return total_tokens, patch_tokens


def compute_stats(
    img_size: int,
    patch_size: int,
    embed_dim: int,
    num_blocks: int,
    mlp_ratio: float,
) -> FlopsStats:
    if embed_dim <= 0:
        raise ValueError("embed_dim must be > 0")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if mlp_ratio <= 0:
        raise ValueError("mlp_ratio must be > 0")

    n_total, n_patch = _num_tokens(img_size, patch_size)
    d = embed_dim
    k = POOL_KERNEL_SIZE
    mlp_hidden_dim = int(round(mlp_ratio * d))

    if mlp_hidden_dim <= 0:
        raise ValueError("mlp_hidden_dim resolved from mlp_ratio must be > 0")

    # Standard ViT MHA token mixer MACs (QKV + output projection + attention matmuls)
    # QKV+Out: 4 * N * d^2, attention matmuls: 2 * N^2 * d
    mha_macs_per_block = 4 * n_total * d * d + 2 * n_total * n_total * d

    # PoolAttention MACs from model/efficient_modules/pool.py
    # patch local avg-pool (k^2 adds) + cls global avg over patches
    pool_macs_per_block = n_patch * d * (k * k + 1)

    # Approximate full ViT backbone MACs/FLOPs (no classifier/projection head):
    # - patch embedding conv
    # - transformer blocks = attention/token mixer + MLP
    # LayerNorm / bias / residual add are ignored as lower-order terms.
    patch_embed_macs = n_patch * d * IN_CHANNELS * patch_size * patch_size
    mlp_macs_per_block = 2 * n_total * d * mlp_hidden_dim

    mha_model_macs_total = patch_embed_macs + num_blocks * (mha_macs_per_block + mlp_macs_per_block)
    pool_model_macs_total = patch_embed_macs + num_blocks * (pool_macs_per_block + mlp_macs_per_block)
    mha_model_flops_total = 2 * mha_model_macs_total
    pool_model_flops_total = 2 * pool_model_macs_total

    model_mac_ratio = mha_model_macs_total / pool_model_macs_total
    model_savings_fraction = 1.0 - (pool_model_macs_total / mha_model_macs_total)

    return FlopsStats(
        tokens_total=n_total,
        tokens_patch=n_patch,
        patch_embed_macs=patch_embed_macs,
        mlp_macs_per_block=mlp_macs_per_block,
        mha_model_macs_total=mha_model_macs_total,
        pool_model_macs_total=pool_model_macs_total,
        mha_model_flops_total=mha_model_flops_total,
        pool_model_flops_total=pool_model_flops_total,
        model_mac_ratio_mha_over_pool=model_mac_ratio,
        model_savings_fraction=model_savings_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute theoretical full-model MACs/FLOPs for MHA vs Pool in ViT."
    )
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--num_blocks", type=int, default=12)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = compute_stats(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        mlp_ratio=args.mlp_ratio,
    )

    print(
        "settings: "
        f"img_size={args.img_size}, patch_size={args.patch_size}, embed_dim={args.embed_dim}, "
        f"num_blocks={args.num_blocks}, mlp_ratio={args.mlp_ratio}, "
        f"pool_kernel_size={POOL_KERNEL_SIZE}, in_channels={IN_CHANNELS}"
    )
    print(
        f"tokens: total={stats.tokens_total} (cls+patch), patch={stats.tokens_patch}"
    )

    print("\nmodel_approx (patch_embed + blocks(token_mixer + mlp)):")
    print(f"  patch_embed_macs={stats.patch_embed_macs}")
    print(f"  mlp_macs_per_block={stats.mlp_macs_per_block}")
    print(f"  mha_model_macs_total={stats.mha_model_macs_total}")
    print(f"  pool_model_macs_total={stats.pool_model_macs_total}")
    print(f"  mha_model_flops_total={stats.mha_model_flops_total}")
    print(f"  pool_model_flops_total={stats.pool_model_flops_total}")
    print(f"  model_mac_ratio_mha_over_pool={stats.model_mac_ratio_mha_over_pool:.4f}x")
    print(f"  model_savings={100.0 * stats.model_savings_fraction:.4f}%")


if __name__ == "__main__":
    main()