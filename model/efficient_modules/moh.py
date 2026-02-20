from torch import nn


def build_attention(original_attention: nn.Module, **_kwargs) -> nn.Module:
    raise NotImplementedError("MoH attention module is not implemented yet.")
