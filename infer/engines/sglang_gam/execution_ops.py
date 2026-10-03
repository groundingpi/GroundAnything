"""Execution-only helpers; no sampling, attention, or checkpoint policy."""
from __future__ import annotations

import torch


def cached_extend_positions(positions, batch, input_ids):
    """Use SGLang's authoritative CPU lengths without synchronizing CUDA.

    A ForwardBatch owns the cache. Prefix, length, device, dtype and target
    length form its key, so chunked prefill and subsequent blocks invalidate
    it. Synthetic graph batches retain the graph-owned position buffer.
    Return None when the CPU metadata contract is unavailable.
    """
    if getattr(batch, '_gam_triton_dllm_graph_capture', False):
        return None
    prefixes = getattr(batch, 'extend_prefix_lens_cpu', None)
    lengths = getattr(batch, 'extend_seq_lens_cpu', None)
    if prefixes is None or lengths is None:
        return None
    # Never accidentally synchronize an incorrectly named CUDA field.
    if any(isinstance(x, torch.Tensor) and x.device.type != 'cpu'
           for x in (prefixes, lengths)):
        return None
    prefixes = tuple(int(x) for x in prefixes)
    lengths = tuple(int(x) for x in lengths)
    target = getattr(batch, 'num_token_non_padded_cpu', None)
    target = int(input_ids.numel()) if target is None else int(target)
    if (not prefixes or len(prefixes) != len(lengths)
            or min(prefixes) < 0 or min(lengths) < 0
            or sum(lengths) != target):
        return None
    key = (prefixes, lengths, positions.device, positions.dtype, target)
    cached = getattr(batch, '_gam_absolute_position_cache', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    result = torch.cat([
        torch.arange(p, p + n, device=positions.device, dtype=positions.dtype)
        for p, n in zip(prefixes, lengths)
    ]).contiguous()
    batch._gam_absolute_position_cache = (key, result)
    return result


def longest_prefix_and_correction(ids, predictions):
    """Exact causal prefix with one fixed-size device-to-host transfer.

    nonzero() synchronizes CUDA to discover a dynamic output size. A sentinel
    reduction determines the same first mismatch, including full acceptance,
    and transfers the keep count and correction together.
    """
    n = ids.numel()
    if n < 1 or predictions.numel() != n or ids.ndim != 1 or predictions.ndim != 1:
        raise ValueError('expected nonempty equal-length token vectors')
    offsets = torch.arange(n - 1, device=ids.device, dtype=torch.long)
    mismatch = predictions[:-1].ne(ids[1:])
    first = torch.cat((torch.where(mismatch, offsets, n - 1), offsets.new_full((1,), n - 1))).amin()
    keep = first + 1
    correction = predictions.gather(0, first.reshape(1)).reshape(())
    values = torch.stack((keep, correction)).tolist()
    return int(values[0]), int(values[1])
