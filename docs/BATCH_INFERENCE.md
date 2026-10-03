# 🗂️ Batch Annotation and Data Generation

Use [`examples/batch_predict.py`](../examples/batch_predict.py) to turn an image collection into a resumable JSONL of grounding predictions. It connects to an already running OpenAI-compatible service and uses Python's standard library plus this repository's GAM parser. The client does not load model weights or require an accelerator.

## ⚡ Start a service

Start the desired service using [Inference](INFERENCE.md) or the [VLM vLLM deployment guide](VLLM.md), then select its endpoint and model ID:

| Model | Service | Base URL | Model ID |
|---|---|---|---|
| GroundAnything (DLM) | SGLang, entropy-guided or self-speculative decoding | `http://127.0.0.1:8101/v1` | `groundinganything` |
| GroundAnything-VLM | SGLang or the vLLM adapter | `http://127.0.0.1:8102/v1` | `groundinganything-vlm` |

The vLLM route supports the autoregressive VLM checkpoint. DLM entropy-guided and self-speculative generation use SGLang. The new VLM vLLM adapter requires accelerator validation in your runtime; the CPU client tests below do not validate the model server.

## 📝 Prepare an input manifest

Create `requests.jsonl`, with one JSON object per line:

```jsonl
{"id":"scene-001-car","image":"images/scene-001.jpg","task":"bbox","prompt":"Locate the target referred to by the following description: the red car."}
{"id":"scene-001-center","image":"images/scene-001.jpg","task":"point","prompt":"Point to the target referred to by the following description: the center of the red car."}
{"id":"scene-002-objects","image":"images/scene-002.jpg","task":"bbox","prompt":"Locate all the instances that match the following categories: person</c>car."}
{"id":"document-001-text","image":"images/document-001.png","task":"bbox","prompt":"OCR task detect all the text in box format."}
```

Each row needs a unique string `id`, a local `image` path, a `prompt`, and a geometric `task` (`bbox` or `point`). Relative image paths are resolved against the manifest's directory. JPEG, PNG and WebP files are supported. Multiple prompts for one image need different IDs.

Prompts determine the semantic task; `task` only selects the output geometry for validation. Use the prompt builders in [`grounding_anything/protocol.py`](../grounding_anything/protocol.py) for referring, category detection, point localization, OCR spotting, layout localization and other supported prompt families. OCR spotting returns text regions with transcription labels; this example does not convert documents into reading-order Markdown or implement a document reconstruction pipeline. Consult the model's task descriptions when selecting a prompt.

## 🚀 Run and resume

From the repository root:

```bash
python3 examples/batch_predict.py \
  --input requests.jsonl \
  --output outputs/annotations.jsonl \
  --base-url http://127.0.0.1:8101/v1 \
  --model groundinganything \
  --run-tag groundanything-checkpoint-v1
```

For GroundAnything-VLM, use `--base-url http://127.0.0.1:8102/v1 --model groundinganything-vlm` and a corresponding checkpoint/run tag.

Run the same command again after an interruption. Resume is automatic: an existing successful request is skipped only when its input ID, image SHA-256, prompt, geometry, endpoint, model ID, token budget and run tag match. Input image contents are checked even when the filename is unchanged. Failed attempts remain in the output and are retried on the next run.

**Change `--run-tag` when changing the checkpoint or server-side decoding settings behind the same endpoint/model ID.** The client cannot inspect a running server's weights. A new output path also starts a separate run. Keep one writer per output file; the example does not coordinate concurrent processes.

Useful options:

| Option | Default | Purpose |
|---|---|---|
| `--max-tokens` | `4096` | Per-image output budget; increase within the server context limit for dense scenes. |
| `--timeout` | `180` seconds | HTTP request timeout. A timeout is recorded as a failed attempt. |
| `--run-tag` | Empty | Your checkpoint revision and decoding configuration identifier. |
| `--api-key-env` | Unset | Name of an environment variable containing the service API key, if required. The key is not written to results. |

The script sends requests sequentially with `temperature=0`, `skip_special_tokens=false`, and `spaces_between_special_tokens=false`. Current release service recipes use one active sequence. For independent accelerator replicas, shard the input manifest and give each worker its own endpoint and output path. Client concurrency does not turn a single-sequence service into model-level batching.

For the DLM, this generic image client uses the active server's decoding method and shared request defaults. It does not inject benchmark-specific task budgets or decoding policies. Use the [evaluation workflow](EVALUATION.md) to reproduce benchmark results, and record your chosen server decoder in `--run-tag` for annotation runs.

## ✅ Inspect outputs before using them as annotations

Each attempt is appended and flushed to disk. Records preserve input identity and request settings, raw output, `finish_reason`, usage, parsing errors, and parsed predictions:

```json
{
  "id": "scene-001-car",
  "status": "success",
  "task": "bbox",
  "finish_reason": "stop",
  "predictions": [{"label": "the red car", "coordinates": [[100, 200, 600, 800]]}]
}
```

This is a shortened illustration; real records also include the prompt, image path/hash, request fingerprint, timestamp, and raw GAM text. Use `status == "success"` to select complete, syntactically valid predictions. A negative match has `coordinates: null`, which is valid and must not be converted to a fabricated box.

Bounding boxes are `[x1, y1, x2, y2]`; points are `[x, y]`. Both use integer coordinates on the **0–999 grid**. For an image of width `W` and height `H`, convert using `x_px = x / 999 * (W - 1)` and `y_px = y / 999 * (H - 1)`. Preserve the normalized original values when exporting to another annotation format.

`invalid_response` means the generation was truncated, malformed, or failed the strict GAM grammar check; its predictions are empty and its raw output is retained. `request_error` covers file, connection and HTTP failures. A run exits with code `1` if any attempted item fails, `0` if all items succeed or were already completed, and `2` for manifest/configuration/log errors. A damaged JSONL tail is reported without overwriting it; preserve and repair that line, or select a new output file.

Syntactic success is not a quality score: check that labels and geometry match the prompt and image before adopting generated annotations. After retries, select successful records by `request_id` and keep the request settings with exported data. Different prompts, model settings or run tags intentionally produce distinct records, even for the same input ID.

## 🧪 Client validation

Run the CPU-only transport and resume tests without model weights:

```bash
python3 -m unittest discover -s tests -p test_batch_predict.py -v
```

These tests use a local fake HTTP service to verify special-token flags, image transport, successful resume, changed-input detection, failure retries and rejection of truncated/malformed outputs. They do not measure model accuracy or accelerator throughput. For benchmark reproduction, use the repository's [evaluation workflow](EVALUATION.md).
