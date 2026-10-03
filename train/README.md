# Training

This module provides data preparation, tokenizer tools, runtime checks, training, and checkpoint saving and resumption.

- `data/`: JSONL row validation and bounding-box/point spatial token formats. Raw-data-to-cache conversion is not included.
- `dlm/`: DLM data collation, mixed losses, and custom training loops. Model definitions are in `../models/dlm/`.
- `launcher/`: consistency checks for models, data, checkpoints, and distributed launch settings.
- `rl/`: optional DLM reinforcement learning. Only RLV2 is enabled; the active implementation, shared code, and disabled branches are kept in separate directories.
- `runtime/`: cache loading, media path handling, and runtime utilities. DLM length and image checks and sequence packing are in `dlm/`.
- `tokenizer/`: spatial token extension and checks for consistency between tokenizer and model vocabularies.

See the [training guide](../docs/TRAINING.md). Run commands from the project root.
