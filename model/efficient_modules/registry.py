from typing import Callable, Dict, Iterable

from torch import nn

from .mha import build_attention as build_mha_attention
from .moh import build_attention as build_moh_attention
from .pool import build_attention as build_pool_attention
from .soft import build_attention as build_soft_attention
from .sra import build_attention as build_sra_attention

AttentionBuilder = Callable[..., nn.Module]


# Static built-in attention builders used by this project.
# Runtime registration is intentionally not exposed.
_ATTENTION_REGISTRY: Dict[str, AttentionBuilder] = {
    "mha": build_mha_attention,
    "moh": build_moh_attention,
    "pool": build_pool_attention,
    "soft": build_soft_attention,
    "sra": build_sra_attention,
}


def available_attention_modules() -> Iterable[str]:
    return sorted(_ATTENTION_REGISTRY.keys())


def validate_attention_module(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in _ATTENTION_REGISTRY:
        options = ", ".join(available_attention_modules())
        raise ValueError(f"Unsupported attention module '{name}'. Available: {options}")
    return normalized


def build_attention_module(name: str, original_attention: nn.Module, **kwargs) -> nn.Module:
    normalized = validate_attention_module(name)
    builder = _ATTENTION_REGISTRY[normalized]
    return builder(original_attention=original_attention, **kwargs)


def apply_attention_module(vit_model: nn.Module, name: str) -> nn.Module:
    normalized = validate_attention_module(name)
    if not hasattr(vit_model, "blocks"):
        raise ValueError("ViT model does not expose transformer blocks via 'blocks'.")

    for block_index, block in enumerate(vit_model.blocks):
        if not hasattr(block, "attn"):
            raise ValueError(f"Transformer block at index {block_index} has no 'attn' module.")
        block.attn = build_attention_module(
            normalized,
            original_attention=block.attn,
            block_index=block_index,
            block=block,
        )

    return vit_model
