# ⚡ vLLM Deployment

This optional integration serves the **autoregressive GroundAnything-VLM checkpoint** through vLLM's Transformers backend. GroundAnything's diffusion checkpoint and its entropy-guided / self-speculative algorithms use the custom SGLang implementation documented in [Inference](INFERENCE.md).

**Validation status:** the adapter has CPU-level command-routing, checkpoint-overlay, and request-contract checks. End-to-end inference on GPU or PPU has not yet been validated for this new route.

## 🛠️ Runtime and installation

Use **Linux x86_64 and Python 3.12** with an accelerator-compatible **Torch and vLLM 0.18.x** runtime already installed. NVIDIA GPUs require a matching GPU runtime; PPU requires the corresponding vendor image and its PPU vLLM build. Setup checks that the selected platform matches the installed build. It does not install the base accelerator runtime.

From the GroundAnything repository root:

```bash
python3 -m pip install -r requirements.txt huggingface_hub
hf download GroundingPI/GroundAnything-VLM --local-dir weights/vlm

# NVIDIA GPU
python3 run.py setup vllm --platform gpu
python3 run.py serve --engine vllm --platform gpu
```

For PPU, replace the final two commands with:

```bash
python3 run.py setup vllm --platform ppu
python3 run.py serve --engine vllm --platform ppu
```

Setup creates **`.venv-vllm`**, inherits the platform engine, and installs the repository's **Transformers 5.7.0 fork** and vLLM-profile dependencies. The SGLang `.venv-serve` environment retains its own dependencies. Setup refuses to overwrite an existing environment; skip setup when already prepared, or choose a fresh project-relative directory with the same `--venv` option on setup and serving.

Download the complete VLM package, including custom Python model code, processor, tokenizer, chat template, configuration, and all weight shards. The launcher validates the `GroundAnythingVLMForConditionalGeneration` architecture and creates a separate compatibility overlay. Weight shards are reused without conversion; the source checkpoint is preserved. A GroundAnything DLM bundle is rejected.

## 🚀 Service settings

| Setting | Default |
|:---|:---|
| Base URL | `http://127.0.0.1:8102/v1` |
| Model ID | `groundinganything-vlm` |
| Precision / execution | BF16 / eager; CUDA Graph disabled |
| Context limit | 16,384 tokens, including image, prompt, and output |
| Tensor parallel size / active sequences | 1 / 1 |
| Accelerator memory utilization | 0.7 |
| Multimodal input | One image; video disabled |

The VLM SGLang service uses the same default port and model ID. Stop it before starting vLLM, or select a different port. Once the server is ready, check its model list in another terminal:

```bash
curl --fail http://127.0.0.1:8102/v1/models
```

Expect `groundinganything-vlm` in the returned list. This checks service readiness; send an image request to inspect actual model output.

## 🖼️ Send one image

Run the following in a second terminal with a local JPEG. This example uses Python's standard library:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("example.jpg").read_bytes()).decode("ascii")
payload = {
    "model": "groundinganything-vlm",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            {"type": "text", "text": "Locate the target referred to by the following description: the red car."},
        ],
    }],
    "temperature": 0,
    "max_tokens": 4096,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
}
request = Request(
    "http://127.0.0.1:8102/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
```

Keep **both** `skip_special_tokens=false` and `spaces_between_special_tokens=false`: GAM output encodes boxes and points as adjacent coordinate tokens on a **0–999** grid. The repository's HTTP client also preserves both options. Use the checkpoint's chat template and processor; do not rename the architecture to a stock Qwen model.

## ⚙️ Configuration and runtime checks

The release configurations are [GPU](../configs/release/vlm_vllm_gpu.yaml) and [PPU](../configs/release/vlm_vllm_ppu.yaml). Copy the matching YAML, edit its `args`, and select it explicitly, for example:

```bash
python3 run.py serve --engine vllm --platform gpu \
  --config configs/release/vlm_vllm_gpu.yaml
```

Supported overrides include model and overlay paths, host, port, model ID, context length, and memory utilization. Keep model and overlay directories inside the repository and select a fresh overlay when switching checkpoints. This release requires **TP=1 and one active sequence**; larger values are rejected by the launcher. BF16, eager execution, the Transformers backend, and the single-image limit are also fixed by this serving profile.

The environment records its platform, imports, and package versions. The serving overlay writes `engine_runtime.json` with the engine command, checkpoint, adapter hash, accelerator, and Transformers source. A stale overlay or incompatible model/runtime fails before the server starts. DLM decoder options cannot be combined with the vLLM engine.

## 📈 Evaluate the VLM endpoint

Use **GAM** mode, **`service_contract: openai`**, and model ID **`groundinganything-vlm`**. With data paths prepared and the service running:

```bash
python3 run.py setup eval
python3 run.py eval --config configs/eval/vlm_vllm.yaml
```

This configuration is an 8-sample smoke test. Follow the [30-benchmark Evaluation Guide](../eval/README.md) for the full task list and `limit: null`, starting from the vLLM VLM configuration and using a fresh run ID. Evaluation consumes the running autoregressive endpoint; it does not enable the DLM entropy-guided or self-speculative algorithms.

See the [vLLM Transformers-backend documentation](https://docs.vllm.ai/en/v0.18.0/models/supported_models/#transformers) and [OpenAI-compatible server reference](https://docs.vllm.ai/en/v0.18.0/serving/openai_compatible_server/) for engine interfaces.

## 🗂️ Batch annotation

For image collections, use the resumable [JSONL batch example and guide](BATCH_INFERENCE.md). The client supports the OpenAI-compatible endpoint and preserves GAM coordinate tokens.

## 🐳 NVIDIA container starting point

If you do not already have a vLLM runtime, the [official vLLM container](https://docs.vllm.ai/en/v0.18.0/deployment/docker/) provides a starting point. This example uses a Linux x86_64 host with Docker, NVIDIA Container Toolkit, and a driver compatible with the image. It is a setup recipe, not an additional accelerator-validation result.

From your cloned repository on the host:

```bash
docker run --rm -it --gpus 'device=0' \
  --network host --shm-size 8g \
  -v "$PWD":/workspace/GroundAnything -w /workspace/GroundAnything \
  --entrypoint bash vllm/vllm-openai:v0.18.0
```

Inside the container, prepare a fresh serving environment:

```bash
python3 -m pip install -r requirements.txt huggingface_hub
hf download GroundingPI/GroundAnything-VLM --local-dir weights/vlm
python3 run.py setup vllm --platform gpu
python3 run.py serve --engine vllm --platform gpu
```

The mounted checkout retains weights, environment files, and outputs. On restart with the same image and mount path, skip installation and run the serving command. The default API binds to loopback; Linux host networking lets host-side clients reach it at the URL above. PPU uses its vendor runtime instead of this NVIDIA image. Keep the model adapter and bundled Transformers fork; a stock `vllm serve` invocation alone does not install them.
