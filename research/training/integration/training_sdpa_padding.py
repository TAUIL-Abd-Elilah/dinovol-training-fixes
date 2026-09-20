"""Experimental, default-off FP32 padded SDPA for FP16 Dinovol training.

Zero-padding attention heads is established prior art. This helper addresses
the measured 54-wide float16 Dinovol inputs, promoting the attention operation
to float32; it does not establish convergence or quality equivalence. All native
autograd operations remain differentiable.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _eligible(query, key, value, attn_mask, dropout_p, enable_gqa):
    if attn_mask is not None or dropout_p != 0.0 or enable_gqa:
        return False
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return False
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "is_compiling") or compiler.is_compiling():
        return False
    tensors = (query, key, value)
    if any(t.device.type != "cuda" or t.dtype != torch.float16
           or t.layout != torch.strided or getattr(t, "is_nested", False)
           or t.ndim != 4 for t in tensors):
        return False
    if any(t.device != query.device or t.dtype != query.dtype
           or t.shape[-1] != 54 or t.stride(-1) != 1 or min(t.shape) <= 0
           for t in tensors):
        return False
    return (query.shape[:2] == key.shape[:2] == value.shape[:2]
            and key.shape[-2] == value.shape[-2])


def _enabled_fused_backends():
    backend = torch.backends.cuda
    return [check for enabled, check in (
        (backend.flash_sdp_enabled, backend.can_use_flash_attention),
        (backend.mem_efficient_sdp_enabled, backend.can_use_efficient_attention),
        (backend.cudnn_sdp_enabled, backend.can_use_cudnn_attention),
    ) if enabled()]


def _has_fused_backend(query, key, value, is_causal, checks):
    params = torch.backends.cuda.SDPAParams(query, key, value, None, 0.0, is_causal, False)
    return any(check(params, debug=False) for check in checks)


def padded_training_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                         is_causal=False, *, scale=None, enable_gqa=False,
                         enabled=False, native=None):
    """Preserve native SDPA unless opt-in padded FP32 enables a fused backend.

    FP16 inputs become FP32 before the 54-to-56 zero extension. Attention runs
    with autocast disabled, retaining the original scale; the two extra value
    channels are sliced and output is cast to the input dtype. Cast/pad/slice
    retain native autograd. No backend is forced. Numerical differences remain.

    Capability exceptions delegate; allocation/kernel execution errors remain
    visible. `native` is an optional callable for isolated benchmark interception
    and normally remains None, which uses the current native PyTorch function.
    """
    if native is None:
        native = F.scaled_dot_product_attention
    options = dict(attn_mask=attn_mask, dropout_p=dropout_p,
                   is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if enabled is True and _eligible(query, key, value, attn_mask, dropout_p, enable_gqa):
        try:
            checks = _enabled_fused_backends()
            already_supported = not checks or _has_fused_backend(query, key, value, is_causal, checks)
        except (AttributeError, TypeError, RuntimeError):
            already_supported = True
        if not already_supported:
            with torch.autocast(device_type=query.device.type, enabled=False):
                q, k, v = (F.pad(t.float(), (0, 2)) for t in (query, key, value))
                try:
                    candidate_supported = _has_fused_backend(q, k, v, is_causal, checks)
                except (AttributeError, TypeError, RuntimeError):
                    candidate_supported = False
                if candidate_supported:
                    return native(q, k, v, attn_mask=None, dropout_p=0.0,
                                  is_causal=is_causal,
                                  scale=54 ** -0.5 if scale is None else scale,
                                  enable_gqa=False)[..., :54].to(query.dtype)
    return native(query, key, value, **options)
