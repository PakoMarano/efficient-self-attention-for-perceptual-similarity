import argparse
from dataclasses import dataclass

"""
Theoretical full-model parameter-count comparison for MHA vs Pool token mixer.

What this script prints:
- full-model approximate parameter counts for a ViT backbone, for both `mha` and `pool`.

Assumptions:
- square image and square patch grid (`img_size % patch_size == 0`);
- MHA attention params per block (with bias):
  qkv + out projection = 4 * d^2 + 4 * d;
- PoolAttention params per block from `model/efficient_modules/pool.py`:
  zero learnable parameters;
- MLP params per block (with bias):
  d*hidden + hidden + hidden*d + d;
- LayerNorm params are included (gamma + beta);
- patch embedding uses RGB input channels (=3) and includes bias.
"""

IN_CHANNELS = 3
USE_ATTN_BIAS = True
USE_MLP_BIAS = True
USE_PATCH_EMBED_BIAS = True


@dataclass
class ParamStats:
    tokens_total: int
    tokens_patch: int
    patch_embed_params: int
    cls_token_params: int
    pos_embed_params: int
    mlp_params_per_block: int
    norm_params_per_block: int
    final_norm_params: int
    shared_backbone_params: int
    mha_model_params_total: int
    pool_model_params_total: int
    model_ratio_mha_over_pool: float
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
    total_tokens = patch_tokens + 1
    return total_tokens, patch_tokens


def _linear_params(in_dim: int, out_dim: int, bias: bool) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)


def compute_stats(
    img_size: int,
    patch_size: int,
    embed_dim: int,
    num_blocks: int,
    mlp_ratio: float,
) -> ParamStats:
    if embed_dim <= 0:
        raise ValueError("embed_dim must be > 0")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be > 0")
    if mlp_ratio <= 0:
        raise ValueError("mlp_ratio must be > 0")
    n_total, n_patch = _num_tokens(img_size, patch_size)
    d = embed_dim
    hidden = int(round(mlp_ratio * d))
    if hidden <= 0:
        raise ValueError("mlp_hidden_dim resolved from mlp_ratio must be > 0")

    # Token mixer params per block
    qkv_params = _linear_params(d, 3 * d, USE_ATTN_BIAS)
    out_proj_params = _linear_params(d, d, USE_ATTN_BIAS)
    mha_params_per_block = qkv_params + out_proj_params
    pool_params_per_block = 0

    mha_params_total = num_blocks * mha_params_per_block
    pool_params_total = num_blocks * pool_params_per_block

    # Shared backbone params (rough ViT approximation)
    patch_embed_params = (
        IN_CHANNELS * patch_size * patch_size * d + (d if USE_PATCH_EMBED_BIAS else 0)
    )
    cls_token_params = d
    pos_embed_params = n_total * d

    fc1_params = _linear_params(d, hidden, USE_MLP_BIAS)
    fc2_params = _linear_params(hidden, d, USE_MLP_BIAS)
    mlp_params_per_block = fc1_params + fc2_params

    # Two LayerNorms per block, each with weight+bias over embed dim
    norm_params_per_block = 4 * d
    final_norm_params = 2 * d

    shared_backbone_params = (
        patch_embed_params
        + cls_token_params
        + pos_embed_params
        + num_blocks * (mlp_params_per_block + norm_params_per_block)
        + final_norm_params
    )

    mha_model_params_total = shared_backbone_params + mha_params_total
    pool_model_params_total = shared_backbone_params + pool_params_total

    model_ratio = mha_model_params_total / max(pool_model_params_total, 1)
    model_savings_fraction = 1.0 - (pool_model_params_total / max(mha_model_params_total, 1))

    return ParamStats(
        tokens_total=n_total,
        tokens_patch=n_patch,
        patch_embed_params=patch_embed_params,
        cls_token_params=cls_token_params,
        pos_embed_params=pos_embed_params,
        mlp_params_per_block=mlp_params_per_block,
        norm_params_per_block=norm_params_per_block,
        final_norm_params=final_norm_params,
        shared_backbone_params=shared_backbone_params,
        mha_model_params_total=mha_model_params_total,
        pool_model_params_total=pool_model_params_total,
        model_ratio_mha_over_pool=model_ratio,
        model_savings_fraction=model_savings_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute theoretical full-model parameter counts for MHA vs Pool in ViT."
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
        f"num_blocks={args.num_blocks}, mlp_ratio={args.mlp_ratio}, in_channels={IN_CHANNELS}, "
        f"attn_bias={USE_ATTN_BIAS}, mlp_bias={USE_MLP_BIAS}, "
        f"patch_embed_bias={USE_PATCH_EMBED_BIAS}"
    )
    print(f"tokens: total={stats.tokens_total} (cls+patch), patch={stats.tokens_patch}")

    print("\nmodel_approx:")
    print(f"  patch_embed_params={stats.patch_embed_params}")
    print(f"  cls_token_params={stats.cls_token_params}")
    print(f"  pos_embed_params={stats.pos_embed_params}")
    print(f"  mlp_params_per_block={stats.mlp_params_per_block}")
    print(f"  norm_params_per_block={stats.norm_params_per_block}")
    print(f"  final_norm_params={stats.final_norm_params}")
    print(f"  shared_backbone_params={stats.shared_backbone_params}")
    print(f"  mha_model_params_total={stats.mha_model_params_total}")
    print(f"  pool_model_params_total={stats.pool_model_params_total}")
    print(f"  model_param_ratio_mha_over_pool={stats.model_ratio_mha_over_pool:.4f}x")
    print(f"  model_savings={100.0 * stats.model_savings_fraction:.4f}%")


if __name__ == "__main__":
    main()