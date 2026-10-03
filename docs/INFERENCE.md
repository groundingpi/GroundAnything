# Inference

The default model service uses SGLang and supports three decoding methods.

## Setup and serving

Prepare matching resources in `weights/base_model/` and `weights/dlm/`, then run:

```bash
python3 run.py setup serve
python3 run.py prepare-model
python3 run.py serve
```

Model preparation writes a new `weights/dlm_bundle/`; source checkpoints remain unchanged. All three decoders use this bundle and expose `http://127.0.0.1:8101/v1` with model ID `groundinganything`.

| Command | Generation method | Evaluation command |
|---|---|---|
| `python3 run.py serve` | Direct block denoising, DecodeV4 (default) | `python3 run.py eval` |
| `python3 run.py serve --decoder causal` | Causal AR baseline | `python3 run.py eval --decoder causal` |
| `python3 run.py serve --decoder speculative` | Greedy self-speculation with parallel causal verification | `python3 run.py eval --decoder speculative` |

`--decoder denoise` explicitly selects the default. Stop the existing service before switching. Evaluation selects request parameters and checks `/server_info` for the actual algorithm; it does not reconfigure a running server. Service recipes are `configs/release/dlm_sglang_<decoder>.yaml`. Evaluation recipes are `configs/eval/dlm.yaml`, `dlm_causal.yaml` and `dlm_speculative.yaml`.

## DecodeV4 request policy

DecodeV4 uses B32, sub-block 4, entropy threshold 0.8 and the canonical task profiles in `infer/decode/configs/task_profiles.json`. It iteratively fills masked positions and commits the completed block to causal KV; that commit does not verify or reject the generated content.

| Task tier | Tasks | Temperature | Top-p | Output budget |
|---|---:|---:|---:|---:|
| Strict single target | 14 | 0 | 1 | 512 |
| Medium, non-OCR | 12 | 0.1 | 0.95 | 4096 |
| Dense, non-OCR | 8 | 0.1 | 0.95 | 8192 |
| Medium OCR | 4 | 0.3 | 0.95 | 4096 |
| Dense OCR | 4 | 0 | 1 | 4096 |

Evaluation injects both the outer sampling parameters and the nested `custom_params.gam_dlm_decode` contract. Strict-single tasks also receive the coordinate/cardinality stop policy. Unknown tasks fail instead of receiving a guessed tier. `max_tokens` overrides are rejected for this evaluation policy so the five-tier budgets remain intact; `limit` only changes the number of evaluated samples.

A generic image client has no benchmark task identity. Such requests use the server's shared entropy settings with greedy inner sampling; they do not automatically receive a benchmark's five-tier or single-target policy. To reproduce benchmark metrics, use the evaluation entrypoint. Advanced clients may pass an explicit `custom_params.gam_dlm_decode` object following `infer/decode/request_contract.py`.

Causal and self-speculative evaluation use greedy anchors, top-p 1 and repetition penalty 1. Self-speculation creates a draft, performs one parallel causal verification, and accepts its longest matching prefix. It is a separate algorithm from DecodeV4 and does not implement general stochastic speculative sampling.

## Runtime

The integrated routes use BF16, Triton attention, eager execution, one GPU per service, one active request and two queued requests. Multiple replicas can use separate visible devices, ports and output directories. Client concurrency 2 queues requests; it does not create a two-request model batch.

`outputs/sglang/<decoder>/engine_runtime.json` records the engine source, algorithm, effective server settings, task-profile digest and package versions. Evaluation records its decoder/profile in `run.json`, logs task generation parameters, and rejects a server/decoder mismatch before launching workers. Shell `GAM_*` and `SGLANG_*` experiment overrides are cleared by the service launcher.

CUDA Graph is disabled in the default recipes.

## Native service

The native Transformers service uses the training environment. Prepare the base model in `weights/base_model/` and the DLM weights in `weights/dlm/`, then run:

```bash
.venv-train/bin/python scripts/run.py configs/release/dlm_serve.yaml
```

Set `env.DLM_INFERENCE_MODE` in that configuration to `speculative` or `causal_cached`. Restart the service after changing the configuration.

## Image prediction

Connect to the default endpoint with the Python client:

```python
from grounding_anything import GroundingAnything

client = GroundingAnything()
result = client.predict("your_image.jpg", "the red car", task="bbox")
print(result.to_dict())
```

Use `task="point"` for point localization. See [examples](../examples/README.md) for visualization and command-line usage. Custom API requests must preserve spatial tokens and their adjacency with `skip_special_tokens=false` and `spaces_between_special_tokens=false`.


## GroundAnything-VLM with SGLang

Place the GroundAnything-VLM release in `weights/vlm/`, then start its causal
SGLang service with `python3 run.py serve --config configs/release/vlm_sglang.yaml`.
It serves model ID `groundinganything-vlm` on port 8102. The DLM decoder
recipes continue to use `weights/dlm_bundle/` and port 8101.

## 🗂️ Batch annotation

For image collections, use the resumable [JSONL batch example and guide](BATCH_INFERENCE.md). The client supports the OpenAI-compatible endpoint and preserves GAM coordinate tokens.

See the [vLLM Deployment Guide](VLLM.md) for platform setup, configuration, and a complete image request.
