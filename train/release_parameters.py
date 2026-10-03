"""Supported training YAML controls; no GPU imports."""
TRAINING_FIELDS = {
    'vlm': set(['add_non_thinking_prefix', 'aligner_lr', 'attn_impl', 'data_seed', 'dataloader_num_workers', 'dataset_num_proc', 'dataset_shuffle', 'ddp_timeout', 'deepspeed', 'freeze_aligner', 'freeze_llm', 'freeze_vit', 'gradient_accumulation_steps', 'gradient_checkpointing', 'learning_rate', 'load_from_cache_file', 'logging_steps', 'loss_scale', 'lr_scheduler_kwargs', 'lr_scheduler_type', 'max_grad_norm', 'max_length', 'max_steps', 'num_train_epochs', 'optimizer', 'packing', 'packing_length', 'packing_num_proc', 'per_device_train_batch_size', 'report_to', 'resume_from_checkpoint', 'save_only_model', 'save_steps', 'save_strategy', 'seed', 'split_dataset_ratio', 'swift_module', 'torch_dtype', 'truncation_strategy', 'tuner_type', 'use_logits_to_keep', 'vit_gradient_checkpointing', 'vit_lr', 'warmup_ratio', 'weight_decay']),
    'dlm': set(['adam_beta1', 'adam_beta2', 'adam_epsilon', 'allocator_empty_cache_steps', 'allocator_snapshot_steps', 'backbone_learning_rate', 'block_annealing', 'block_size', 'casuallossenable', 'causal_loss_weight', 'compiler_cache_reset_steps', 'data_seed', 'dataloader_num_workers', 'disable_checkpointing', 'expected_global_batch_size', 'fixed_block_size_experiment', 'freeze_projector', 'freeze_vision_encoder', 'gradient_accumulation_steps', 'gradient_checkpointing', 'language_checkpoint_stride', 'learning_rate', 'logging_steps', 'lr_scheduler_kwargs', 'lr_scheduler_type', 'max_grad_norm', 'max_steps', 'mdm_loss_weight', 'minimum_noise_level', 'num_train_epochs', 'optim', 'optimizer_foreach', 'packing_activation_cpu_offload', 'packing_language_checkpoint_stride', 'packing_linear_kernel', 'packing_lm_head_loss_backend', 'packing_lm_head_loss_chunk_tokens', 'packing_offload_layer_stride', 'packing_offload_min_tokens', 'packing_offload_pin_memory', 'packing_offload_threshold_mib', 'padding_free_packing', 'per_device_train_batch_size', 'projector_learning_rate', 'save_final_full_checkpoint', 'save_steps', 'save_strategy', 'seed', 'stop_after_step', 'training_profile', 'validated_large_batch_size', 'vision_attn_implementation', 'vision_gradient_checkpointing', 'vision_learning_rate', 'warmup_ratio', 'weight_decay']),
}

def validate_training_fields(config):
    """Reject unknown training parameters; metadata is explicitly informational."""
    unknown = set(config) - {'mode', 'meta', 'model', 'data', 'training', 'runtime', 'datasets'}
    if unknown:
        raise ValueError(f'unknown native configuration fields: {sorted(unknown)}')
    training = config.get('training', {})
    if not isinstance(training, dict): raise ValueError('training must be a mapping')
    family = 'dlm' if config.get('mode') == 'DLM' else 'vlm'
    unknown = set(training) - TRAINING_FIELDS[family]
    if unknown: raise ValueError(f'unknown {family} training fields: {sorted(unknown)}')
    for key in ('max_steps', 'stop_after_step'):
        if key in training:
            value = training[key]
            if type(value) is not int or (value < 1 and not (key == 'max_steps' and value == -1)):
                raise ValueError(f'training.{key} must be positive (max_steps also accepts -1)')
    if 'stop_after_step' in training:
        if not 0 < training['stop_after_step'] < training.get('max_steps', -1):
            raise ValueError('stop_after_step requires 0 < stop_after_step < max_steps')
    for key in ('disable_checkpointing', 'dataset_shuffle', 'load_from_cache_file'):
        if key in training and type(training[key]) is not bool:
            raise ValueError(f'training.{key} must be a YAML boolean')
    if 'swift_module' in training and training['swift_module'] != 'sft':
        raise ValueError('vlm-train supports only swift_module=sft')


def dlm_control_args(config):
    validate_training_fields(config)
    result = []
    for key in ('max_steps', 'stop_after_step'):
        if key in config.get('training', {}):
            result.extend(['--' + key.replace('_', '-'), str(config.get('training', {})[key])])
    if 'disable_checkpointing' in config.get('training', {}):
        result.append('--disable-checkpointing' if config.get('training', {})['disable_checkpointing']
                      else '--no-disable-checkpointing')
    return result


def resolve_dlm_controls(args, config):
    """Apply YAML before schedule checks; conflicting CLI controls are errors."""
    validate_training_fields(config)
    for key, default in (('max_steps', -1), ('stop_after_step', None), ('disable_checkpointing', False)):
        cli = getattr(args, key)
        training = config['training']
        if key in training and cli is not None and training[key] != cli:
            raise ValueError(f'CLI {key} conflicts with training.{key}')
        setattr(args, key, training.get(key, default) if cli is None else cli)
    if args.stop_after_step is not None and not 0 < args.stop_after_step < args.max_steps:
        raise ValueError('stop_after_step must be inside the max_steps schedule')
    return args
