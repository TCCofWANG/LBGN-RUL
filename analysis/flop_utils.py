

import math
import re
from contextlib import contextmanager

import torch
import thop


def _einsum_macs(equation, operands):

    if "->" in equation:
        lhs, _ = equation.split("->")
    else:
        lhs = equation
    in_specs = [s.strip() for s in lhs.split(",")]
    if len(in_specs) != 2 or len(operands) != 2:


        raise NotImplementedError(
            f"_einsum_macs only handles 2-operand einsums (got equation={equation!r}, "
            f"{len(operands)} operands) -- extend this function before trusting its output."
        )
    sizes = {}
    for spec, op in zip(in_specs, operands):
        spec = spec.replace("...", "")
        for letter, dim in zip(spec, op.shape):
            sizes[letter] = int(dim)
    macs = 1
    for v in sizes.values():
        macs *= v
    return macs


def _matmul_macs(a, b):

    a_shape, b_shape = a.shape, b.shape
    m, k = a_shape[-2], a_shape[-1]
    k2, n = b_shape[-2], b_shape[-1]
    assert k == k2, f"matmul shape mismatch: {a_shape} @ {b_shape}"
    a_batch = list(a_shape[:-2])
    b_batch = list(b_shape[:-2])
    ndim = max(len(a_batch), len(b_batch))
    a_batch = [1] * (ndim - len(a_batch)) + a_batch
    b_batch = [1] * (ndim - len(b_batch)) + b_batch
    batch = 1
    for da, db in zip(a_batch, b_batch):
        batch *= max(da, db)
    return int(batch) * int(m) * int(k) * int(n)


def _rfft_macs(x, n_transformed):

    n = n_transformed
    if n <= 1:
        return 0
    per_transform = 5.0 * n * math.log2(n)
    n_slices = x.numel() // n
    return per_transform * n_slices


@contextmanager
def _patched_ops(counters):

    orig_einsum = torch.einsum
    orig_matmul = torch.matmul
    orig_mm = torch.mm
    orig_tensor_matmul = torch.Tensor.__matmul__
    orig_rfft = torch.fft.rfft

    def patched_einsum(equation, *operands):
        counters["einsum_matmul"] += _einsum_macs(equation, operands)
        return orig_einsum(equation, *operands)

    def patched_matmul(a, b, *args, **kwargs):
        counters["einsum_matmul"] += _matmul_macs(a, b)
        return orig_matmul(a, b, *args, **kwargs)

    def patched_mm(a, b, *args, **kwargs):
        counters["einsum_matmul"] += _matmul_macs(a, b)
        return orig_mm(a, b, *args, **kwargs)

    def patched_tensor_matmul(self, other):
        counters["einsum_matmul"] += _matmul_macs(self, other)
        return orig_tensor_matmul(self, other)

    def patched_rfft(x, n=None, dim=-1, norm=None):
        transform_len = n if n is not None else x.shape[dim]
        counters["fft"] += _rfft_macs(x, transform_len)
        return orig_rfft(x, n=n, dim=dim, norm=norm)

    torch.einsum = patched_einsum
    torch.matmul = patched_matmul
    torch.mm = patched_mm
    torch.Tensor.__matmul__ = patched_tensor_matmul
    torch.fft.rfft = patched_rfft
    try:
        yield counters
    finally:
        torch.einsum = orig_einsum
        torch.matmul = orig_matmul
        torch.mm = orig_mm
        torch.Tensor.__matmul__ = orig_tensor_matmul
        torch.fft.rfft = orig_rfft


def count_flops(model, forward_args):

    model.eval()
    counters = {"einsum_matmul": 0, "fft": 0.0}
    with torch.no_grad(), _patched_ops(counters):
        thop_macs, thop_params = thop.profile(model, inputs=forward_args, verbose=False)

    return {
        "thop_layer_macs": int(thop_macs),
        "einsum_matmul_macs": int(counters["einsum_matmul"]),
        "fft_macs_estimated": float(counters["fft"]),
        "total_macs_excl_fft": int(thop_macs) + int(counters["einsum_matmul"]),
        "total_macs_incl_fft": int(thop_macs) + int(counters["einsum_matmul"]) + int(round(counters["fft"])),
        "params": int(thop_params),
    }
