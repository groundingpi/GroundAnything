# Data Preparation

## Models and caches

Place the base model, tokenizer, processor, and model code in `weights/base_model/`. For inference or checkpoint initialization, place the complete DLM wrapper weights and matching tokenizer in `weights/dlm/`.

Training uses Arrow caches prepared with the matching tokenizer and processor. Each cache requires a `cache_manifest.json`. A generic JSONL-to-cache converter is not included.

Source JSONL rows contain `id`, `messages`, `images`, and `source`. Each `<image>` placeholder corresponds to an image. Spatial answers use object-reference and box markers, coordinate tokens `<0>` through `<999>`, and `</c>`. A bounding box contains four coordinates and a point contains two. DLM checkpoints also require the atomic mask token `|<MASK>|`.

For a base model with an expanded spatial vocabulary, generate its tokenizer manifest:

```bash
.venv-train/bin/python -m train.tokenizer.validate_tokens weights/base_model \
  --write-manifest weights/base_model/gam_tokenizer_manifest.json
```

## Select training data

Declare your prepared caches in a YAML file, for example `data/sources.yaml`:

```yaml
datasets:
  - name: spatial_train
    source_id: spatial_train
    path: data/spatial_train/cache/train
    manifest: data/spatial_train/cache/cache_manifest.json
```

The source ID and preprocessing settings must match the cache manifest. Use `train.dlm.build_sampling_manifest` to select rows and generate their index files. Set `--sample-size` to the `data.sampled_rows` value in the chosen training configuration.

For the supplied general-stage configuration:

```bash
.venv-train/bin/python -m train.dlm.build_sampling_manifest \
  --config data/sources.yaml --output-dir data/general/sampling \
  --sample-size 2997407 --seed 32 --stage general
.venv-train/bin/python -m train.dlm.build_selected_lengths \
  --sampling-manifest data/general/sampling/manifest.json \
  --output data/general/selected_lengths.npy --workers 8
```

Adjust the sample count and source list to your dataset, and update the training configuration accordingly. Sampling creates indices without changing the source cache.

## Check lengths and images

Generate the following inputs from the same selected rows, model, and tokenizer:

| Training field | Preparation tools |
|---|---|
| `runtime.selected_tail_actual_length_audit` | `train.dlm.audit_selected_actual_lengths`, then `train.dlm.materialize_pass_actual_length_workload` |
| `runtime.selected_image_integrity_audit` | `train.dlm.audit_selected_images` |
| `runtime.selected_response_block_audit` | `train.dlm.audit_response_block_lengths`, then `train.dlm.summarize_response_block_distribution` |

Each module exposes its arguments with `--help`, for example:

```bash
.venv-train/bin/python -m train.dlm.audit_selected_actual_lengths --help
```

Length checks must use the training configuration's model family, image token budget, `max_length`, and `max_joint_length`. Use `--min-cached-length 0` to check all selected rows and `--include-multi-image` to include multi-image examples. The derived length report links the selected rows to their measured clean and combined lengths.

Image checks decode the selected images and support sharding; merge shard reports before training. Response checks measure supervised token lengths with the training tokenizer. Generate the response summary from both the measured token array and its parent report, using the configured `minimum_over_32_response_ratio`.

Set the three fields above to the generated reports. When data or preprocessing changes, regenerate the affected inputs.

## Packing

When `training.padding_free_packing` is enabled, create a packing manifest matching the batch size, world size, epoch count, and measured workload lengths:

```bash
.venv-train/bin/python -m train.dlm.build_packing_manifest \
  --sampling-manifest data/general/sampling/manifest.json \
  --selected-lengths data/general/workload_lengths.npy \
  --output-dir data/general/packing --epochs 2 --world-size 64 \
  --pack-size 32 --packing-strategy balanced --seed 32
```

Here `data/general/workload_lengths.npy` is the `--output-workload-lengths` result from the length preparation step. Bind the generated manifest to `data.packing_manifest`. The values shown match the supplied general-stage topology; keep them consistent if you customize the configuration.

See [Training](TRAINING.md) for launch and checkpoint options.

## Optional RL data

RLV2 uses JSONL data. For the 24-device configuration, set these fields in `configs/rl/default.yaml`:

```yaml
data:
  grounding_path: data/rl/grounding.jsonl
  ocr_path: data/rl/ocr.jsonl
```

The current two-source recipe requires 6,557 Grounding rows and 5,500 OCR rows with unique IDs. It selects 4,400 OCR rows deterministically and combines them with all Grounding rows. The row fields and prompt formats are defined in `train/rl/data.py` and `train/rl/corrected_data.py`.

The 56- and 64-device configurations use a multiroute source:

```yaml
data:
  multiroute_path: data/rl/multiroute.jsonl
```

This recipe requires 17,700 rows, unique IDs, the complete route set, and per-row metadata. The `rlv3_route` and `rlv3_audit` field names belong to the shared data schema. See `train/rl/shared/causal_data.py` for the field definitions.

The launcher reads dataset paths from the `data` mapping. Training writes `stable_gate.json` after the first successful update and `final_gate.json` when it finishes, under `outputs/rl/` by default.

## Evaluation data

Configure evaluation annotations and image locations in `configs/datasets.yaml`. These paths can be relative to the evaluation recipe's `data_root` or absolute. See [Evaluation](EVALUATION.md#data-and-tasks).
