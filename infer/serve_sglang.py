"""Launch the bundled SGLang engine with the Kimi/GroundAnything DLM plugins."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_launch(model, output, decoder, mask, vocab, eos, paths, inherited,
                 host='127.0.0.1', port=8101, served_model_name='groundinganything'):
    """Build the exact command/environment without loading CUDA or writing files."""
    from infer.decoding import decoder_settings
    settings = decoder_settings(decoder)
    algorithm_config = output / (decoder + '.yaml')
    # A shell left over from an experiment must not enable no-verify, Graph,
    # quantization or a different sampling policy behind the selected route.
    env = {k: v for k, v in inherited.items()
           if not k.startswith(('GAM_', 'SGLANG_'))}
    env.update(PYTHONPATH=os.pathsep.join(paths), SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_gam",
               SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="sglang_gam",
               SGLANG_EXTERNAL_MM_MODEL_ARCH="Fast_dVLMForConditionalGeneration",
               GAM_SGLANG_MODEL_CODE_PATH=str(model), GAM_SGLANG_MASK_ID=str(mask),
               GAM_SGLANG_IM_END_TOKEN_ID=str(eos),
               GAM_SGLANG_VOCAB_SIZE=str(vocab), GAM_SGLANG_LOGITS_VOCAB_SIZE=str(mask),
               GAM_SGLANG_BLOCK_SIZE="32", GAM_SGLANG_COMPAT="1",
               GAM_SGLANG_ALGORITHM="decode_v2" if decoder == "denoise" else decoder,
               GAM_SGLANG_VISION_ATTN="eager", GAM_SGLANG_DISABLE_CUDA_GRAPH="1",
               SGLANG_ENABLE_JIT_DEEPGEMM="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    if decoder == 'speculative':
        env.update(GAM_SGLANG_SPEC_TRACE_PATH=str(output / 'speculative-trace.jsonl'),
                   GAM_SGLANG_SPEC_TRACE_FLUSH_EVERY='1')
    if decoder == 'denoise':
        env.update(GAM_DLM_DECODE_PROFILE='task_profiles', GAM_DLM_SINGLE_TARGET_RULE='1',
                   GAM_DLM_DECODE_TELEMETRY_MODE='off')
    # This SGLang snapshot initializes its optional classify handler even for
    # generative models. Supply a runtime-only label so startup succeeds; this
    # does not change the released checkpoint or enable classification.
    command = [sys.executable, "-m", "infer.sglang_server", "--model-path", str(model),
               "--served-model-name", served_model_name, "--host", host, "--port", str(port),
               "--device", "cuda", "--dtype", "bfloat16", "--tensor-parallel-size", "1",
               "--trust-remote-code", "--enable-multimodal", "--json-model-override-args",
               '{"architectures":["Fast_dVLMForConditionalGeneration"],"id2label":{"0":"LABEL_0"}}', "--skip-server-warmup",
               "--chat-template", str(model / "chat_template.jinja"), "--attention-backend", "triton",
               "--sampling-backend", "pytorch", "--grammar-backend", "none",
               "--mem-fraction-static", "0.65", "--max-total-tokens", "32768",
               "--max-running-requests", "1", "--max-queued-requests", "2",
               "--disable-radix-cache", "--chunked-prefill-size", "16384",
               "--disable-cuda-graph", "--disable-overlap-schedule"]
    if settings['algorithm']:
        command += ['--dllm-algorithm', settings['algorithm'], '--dllm-algorithm-config', str(algorithm_config)]
    return command, env, settings, algorithm_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="prepared DLM model bundle inside the project")
    p.add_argument("--decoder", choices=("denoise", "causal", "speculative"), default="denoise")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--served-model-name", default="groundinganything")
    p.add_argument("--output", default="outputs/sglang")
    a = p.parse_args()
    model, output = [(ROOT / x).resolve() for x in (a.model, a.output)]
    if any(not x.is_relative_to(ROOT) for x in (model, output)):
        raise ValueError("model and output must remain inside the project")
    if model == output or model in output.parents or output in model.parents:
        raise ValueError("output must not overlap the model bundle")
    from models.dependency_contract import verify_dependency
    revision = verify_dependency("sglang")
    vendor = ROOT / "vendor/sglang"
    paths = [str(vendor), str(ROOT / "infer/engines"), str(ROOT)]
    sys.path[:0] = paths
    import sglang
    if not Path(sglang.__file__).resolve().is_relative_to(vendor.resolve()):
        raise RuntimeError("SGLang did not load the bundled custom source")
    from transformers import AutoTokenizer
    from safetensors import safe_open
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    mask = tokenizer.encode("|<MASK>|", add_special_tokens=False)
    if len(mask) != 1 or tokenizer.convert_ids_to_tokens(mask[0]) != "|<MASK>|":
        raise ValueError("DLM bundle is missing the atomic mask token")
    with safe_open(model / "model.safetensors", framework="pt") as weights:
        vocab = weights.get_slice("base_model.model.language_model.embed_tokens.weight").get_shape()[0]
    if vocab != len(tokenizer) or mask[0] != vocab - 1:
        raise ValueError("DLM tokenizer does not match the checkpoint embedding rows")
    command, env, settings, algorithm_config = build_launch(
        model, output, a.decoder, mask[0], vocab,
        tokenizer.convert_tokens_to_ids('<|im_end|>'), paths, os.environ,
        a.host, a.port, a.served_model_name)
    output.mkdir(parents=True, exist_ok=True)
    if settings['algorithm_config'] is not None:
        import yaml
        algorithm_config.write_text(yaml.safe_dump(settings['algorithm_config']))
    evidence = dict(engine="sglang", revision=revision, engine_file=sglang.__file__,
                    decoding=settings, execution="eager", decoder=a.decoder, checkpoint=str(model.relative_to(ROOT)), command=command,
                    packages={n: importlib.metadata.version(n) for n in
                              ("sglang", "torch", "transformers", "sgl-kernel", "triton", "flashinfer-python")},
                    plugins={k: v for k, v in env.items() if k.startswith("SGLANG_EXTERNAL_")})
    (output / "engine_runtime.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
