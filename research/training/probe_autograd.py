"""Bounded gradient correctness screen before the full training experiment."""
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from training_sdpa import training_sdpa


def compare(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return {"max_abs": float((a-b).abs().max()),
            "mean_abs": float((a-b).abs().mean()),
            "relative_l2": float(torch.linalg.vector_norm(a-b) /
                                 torch.linalg.vector_norm(a).clamp_min(1e-30)),
            "cosine": float(F.cosine_similarity(a, b, dim=0)),
            "finite": bool(torch.isfinite(b).all())}


def run(device, dtype, shape, math_only):
    torch.manual_seed(20620)
    original = [torch.randn(shape, dtype=dtype, device=device) for _ in range(3)]
    upstream = torch.randn(shape, dtype=dtype, device=device) * .01
    results = []
    for func in (F.scaled_dot_product_attention, training_sdpa):
        leaves = [t.detach().clone().requires_grad_() for t in original]
        context = sdpa_kernel(SDPBackend.MATH) if math_only else sdpa_kernel(
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION,
             SDPBackend.CUDNN_ATTENTION])
        with context:
            output = func(*leaves)
            (output * upstream).sum().backward()
        results.append([output.detach().cpu()] + [t.grad.cpu() for t in leaves])
        del output, leaves
    return {"shape": shape, "dtype": str(dtype), "device": device,
            "math_only": math_only,
            "comparisons": {name: compare(a, b) for name, a, b in
                zip(["output", "grad_q", "grad_k", "grad_v"], *results)}}


if __name__ == "__main__":
    torch.set_num_threads(2)
    started = time.time()
    report = {"torch": torch.__version__, "cases": []}
    report["cases"].append(run("cpu", torch.float64, (2, 3, 31, 54), True))
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.85)
        report["gpu"] = torch.cuda.get_device_name()
        report["cases"].append(run("cuda", torch.float16, (2, 16, 4097, 54), False))
        report["cases"].append(run("cuda", torch.bfloat16, (2, 16, 4097, 54), False))
    report["seconds"] = time.time()-started
    p = Path(__file__).with_name("autograd_screen.json")
    p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
