# Model Definitions

- `vlm/`: VLM models, configuration, and processor.
- `vlm_compat.py`: checkpoint loading and inference compatibility utilities.
- `dlm/`: DLM wrapper models shared by training and inference.
- `dependency_contract.py`: dependency source integrity checks.

Serialized class names and configuration identifiers in existing checkpoints are used during model loading and must match the model code. `scripts/prepare_model_code.py` checks model directories and adds missing model code files.
