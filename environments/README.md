# Environment Setup

Run commands from the repository root. The root `requirements.txt` installs the HTTP client, visualization and setup dependencies declared in `pyproject.toml`:

```bash
python -m pip install -r requirements.txt
```

## Optional Conda environment

Use `environment.yml` to create a Python environment for the client and setup commands:

```bash
conda env create -f environment.yml
conda activate grounding-anything
python -m pip install -r requirements.txt
```

The Conda file provides Python and pip. Accelerator libraries are installed or supplied separately by the workflow setup below; for workflows that reuse vendor packages, run setup with the matching platform image's Python interpreter.

## Workflow dependencies

Training, serving, and evaluation use separate environments because their framework versions differ:

| Profile | Package list | Setup command |
|---|---|---|
| train | [requirements/train.txt](../requirements/train.txt) | `python3 run.py setup train` |
| serve | [requirements/serve.txt](../requirements/serve.txt) | `python3 run.py setup serve` |
| eval | [requirements/eval.txt](../requirements/eval.txt) | `python3 run.py setup eval` |

Use these setup commands to install a complete workflow. Installing a profile's package list alone does not prepare the bundled custom frameworks or apply the required installation order.

The installer reads `install.yaml` for source selection, platform checks and build steps, verifies the dependency archives, extracts the required source into `vendor/`, and creates a new virtual environment. Package constraints live in `requirements/`. The installer refuses to overwrite an existing environment; use `--venv` to select another directory and `--index-url` to select a package mirror.

For a setup preview, run `python3 run.py setup PROFILE --dry-run`. To see the full installation plan without applying it, use `python3 scripts/setup_environment.py --profile PROFILE --venv .venv-PROFILE`. Add `--apply` to install through this lower-level script. The interpreter used for workflow setup is provided by `environment.yml` or the matching platform image.

The installer records the resolved packages in each environment's `installed.freeze.txt`.

## Accelerator setup

Training uses the H800 recipe with the CUDA build toolchain specified in `install.yaml`; FlashAttention is built from source. Serving installs the bundled custom SGLang engine, defaults to DecodeV4 denoising, and also supports causal and self-speculative decoding. Default serving uses Triton attention with CUDA Graph disabled.

The serving recipe explicitly installs cuDNN 9.16 for SGLang compatibility. This differs from Torch 2.9.1's package-metadata pin; `serve-dependency-check.json` records that known conflict and rejects others.

Evaluation runs independently of the accelerator environment and connects to an already running service. See [Training](../docs/TRAINING.md), [Inference](../docs/INFERENCE.md), and [Evaluation](../docs/EVALUATION.md) for the workflows.
