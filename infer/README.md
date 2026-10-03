# Inference

This module provides request formats, dependency checks, model code integration, and model services.

- `decode/`: DecodeV4 request parameters, task tiers, and structural constraints, with request formats shared by the evaluation client.
- `dlm/`: the native DLM HTTP service, causal/speculative/hierarchy decoding, and cache implementations.
- `engines/`: custom SGLang model, sampling, and execution adapters used by the default service.

VLM definitions and their original license are in `../models/vlm/`; the DLM wrapper is in `../models/dlm/`. Model weights are supplied separately.

See the [inference guide](../docs/INFERENCE.md). Run commands from the project root.
