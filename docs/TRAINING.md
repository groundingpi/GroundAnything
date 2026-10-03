# Training

GroundAnything uses a custom DLM trainer. Prepare the base model and training inputs as described in [Data Preparation](DATA_PREPARATION.md), then install the environment:

```bash
python3 run.py setup train
```

## Training stages

The stage configurations provide a sequence from general training to spatial specialization and supervised fine-tuning:

| Stage | Training configuration | Launch configuration |
|---|---|---|
| General | `configs/train/general.yaml` | `configs/release/general_train.yaml` |
| Specialist | `configs/train/specialist.yaml` | `configs/release/specialist_train.yaml` |
| SFT 1 | `configs/train/sft1.yaml` | `configs/release/sft1_train.yaml` |
| SFT 2 | `configs/train/sft2.yaml` | `configs/release/sft2_train.yaml` |

Set the model, data, output, batch, and sequence-length settings in the training configuration. Set the distributed topology in the launch configuration. The supplied stage templates use eight nodes with eight devices each; configure the coordinator address and node rank for your cluster before starting them.

Run the selected stage from the project root on each node:

```bash
python3 run.py train --config configs/release/general_train.yaml
```

For a command preview:

```bash
.venv-train/bin/python scripts/run.py configs/release/general_train.yaml --dry-run
```

Outputs are written to `model.output_dir` in the training configuration. The general stage defaults to `outputs/general/`.

## Data and distributed settings

Training consumes a sampling manifest, length and image checks, and response-length statistics. Packed training additionally requires a packing manifest generated for the configured batch size, world size, and number of epochs. See [Data Preparation](DATA_PREPARATION.md).

Set `distributed.nnodes`, `nproc_per_node`, `node_rank`, `master_addr`, and `master_port` for each node. Keep the runtime topology and `training.expected_global_batch_size` consistent. Changes to packed training topology require a matching packing manifest.

## Continue from a checkpoint

For a new stage, set `checkpoint.initial_dlm_checkpoint` in its launch configuration to the preceding stage's DLM weights. This starts a new optimizer and scheduler.

To resume an interrupted stage, set `checkpoint.resume_from_checkpoint` to its complete training checkpoint. This restores optimizer, scheduler, and training state. The two checkpoint options are mutually exclusive.

## Optional reinforcement learning

RLV2 uses a separate training loop:

```bash
.venv-train/bin/python scripts/run.py configs/release/rl_train.yaml
```

Configure the algorithm and data paths in `configs/rl/default.yaml`, and the model, output, and topology in the launch YAML. The supplied launch configuration uses 24 devices. The `rl-56` and `rl-64` entrypoints use a multiroute dataset and their corresponding world sizes.

See [RL data preparation](DATA_PREPARATION.md#optional-rl-data) and the [RL module](../train/rl/README.md) for data fields and available implementations.
