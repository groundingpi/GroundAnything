# Optional Reinforcement Learning

Only RLV2 (causal JustGRPO) is enabled. The launch configuration is `configs/release/rl_train.yaml`, and algorithm parameters are in `configs/rl/default.yaml`. The default configuration uses 24 devices.

| Directory | Contents |
|---|---|
| `current/` | Active trainer, configuration, rollouts, and trajectories. |
| `shared/` | Multi-source data, routing, rewards, and SGLang kernel compatibility utilities. |
| `distributed/` | Adapters for 24-, 56-, and 64-device configurations; `trace` provides shared code for retained branches. |
| `rewards/` | Rewards for grounding, OCR, and other tasks. |
| `branches/trace/`, `corrected/`, `same_data/` | RLV1 and its corrected and comparison branches. |
| `branches/dare/` | The RLV3 DARE branch, checkpoint conversion, and runtime adapters. |
| `branches/rebalanced/`, `rebalanced_64/` | Disabled data-rebalancing comparison branches. |

Trainers under `branches/` and `distributed/trace.py` reject direct execution, but can be imported to reuse shared code. Inspection and conversion tools remain available. Launch configurations support only RLV2 topology adapters; retaining other branches does not enable their algorithms.

Model and data fields must follow the trainer's input format. See [data preparation](../../docs/DATA_PREPARATION.md#optional-rl-data).
