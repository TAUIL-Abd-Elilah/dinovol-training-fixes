"""Research-only SDPA zero-padding adapter, including native autograd.

This is an experiment, not an extension of the published inference-only helper.
The caller explicitly installs it in an isolated benchmark process. Padding an
attention head is established prior art; this experiment tests its application
to the official Dinovol training step on a Windows RTX 3090.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

_NATIVE_SDPA = F.scaled_dot_product_attention


def training_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, *, scale=None, enable_gqa=False,
                  native=_NATIVE_SDPA):
    """Pad only unmasked, zero-dropout Dinovol 54-wide attention to width 56.

    Torch's ordinary pad/slice autograd handles derivatives. The attention scale
    is based on the original query width. Unsupported cases use native SDPA.
    Backend choice is left to PyTorch (or the benchmark's explicit math control).
    """
    options = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                   is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    tensors = (query, key, value)
    eligible = (
        attn_mask is None and dropout_p == 0.0 and not enable_gqa
        and all(t.layout == torch.strided and not getattr(t, "is_nested", False)
                and t.ndim == 4 and t.shape[-1] == 54 and t.stride(-1) == 1
                and t.device == query.device and t.dtype == query.dtype
                for t in tensors)
        and query.shape[:2] == key.shape[:2] == value.shape[:2]
        and key.shape[-2] == value.shape[-2]
    )
    if not eligible:
        return native(query, key, value, **options)
    q, k, v = (F.pad(t, (0, 2)) for t in tensors)
    return native(q, k, v, dropout_p=0.0, is_causal=is_causal,
                  scale=54 ** -0.5 if scale is None else scale)[..., :54]
