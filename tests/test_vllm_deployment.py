"""CPU-only regression checks for VLM deployment; no server or weights are loaded."""

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer.overlay.prepare_vllm_overlay import PATCHERS, create_overlay, source_files, verify_overlay
from infer.serve_vllm import build_command, parser_for
from scripts.setup_vllm_environment import environment_plan

spec = importlib.util.spec_from_file_location("source_tree_cli", ROOT / "run.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def plan(*arguments, root=ROOT):
    args = cli.parser_for(root).parse_args([*arguments, "--dry-run"])
    return cli.plan(args, root)


def make_checkpoint(root):
    model = root / "weights/vlm"
    model.mkdir(parents=True)
    config = {
        "architectures": ["GroundAnythingVLMForConditionalGeneration"],
        "model_type": "groundinganything_vlm", "text_config": {"model_type": "qwen3"},
    }
    (model / "config.json").write_text(json.dumps(config))
    for name in PATCHERS:
        shutil.copyfile(ROOT / "models/vlm" / name, model / name)
    for name in ("image_processing_groundinganything.py", "media_utils.py"):
        shutil.copyfile(ROOT / "models/vlm" / name, model / name)
    for name in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (model / name).write_text("{}")
    # This is a file-link fixture, deliberately not a loadable tensor checkpoint.
    (model / "model.safetensors").write_bytes(b"test-weight-link-only")
    return model


class CliTests(unittest.TestCase):
    def test_existing_sglang_routes_are_preserved(self):
        for decoder in (None, "denoise", "causal", "speculative"):
            extra = [] if decoder is None else ["--decoder", decoder]
            selected = plan("serve", *extra)
            self.assertEqual(selected["profile"], "serve")
            self.assertIn(".venv-serve", selected["command"][0])
            self.assertEqual(selected["command"][-1], f"configs/release/dlm_sglang_{decoder or 'denoise'}.yaml")
        selected = plan("serve", "--config", "configs/release/vlm_sglang.yaml")
        self.assertEqual(selected["profile"], "serve")
        self.assertTrue(plan("setup", "serve")["command"][1].endswith("setup_environment.py"))
        self.assertEqual(plan("eval")["command"][-1], "configs/eval/dlm.yaml")

    def test_vllm_platform_routes_use_separate_environment(self):
        for platform in ("gpu", "ppu"):
            selected = plan("serve", "--engine", "vllm", "--platform", platform)
            self.assertEqual(selected["profile"], "vllm")
            self.assertIn(".venv-vllm", selected["command"][0])
            self.assertEqual(selected["command"][-1], f"configs/release/vlm_vllm_{platform}.yaml")
            selected = plan("setup", "vllm", "--platform", platform)
            self.assertTrue(selected["command"][1].endswith("setup_vllm_environment.py"))
            self.assertIn(platform, selected["command"])
            selected = plan("serve", "--config", f"configs/release/vlm_vllm_{platform}.yaml")
            self.assertEqual(selected["profile"], "vllm")

    def test_incompatible_route_options_fail(self):
        invalid = [
            ("serve", "--engine", "vllm", "--platform", "gpu", "--decoder", "denoise"),
            ("serve", "--platform", "gpu"),
            ("setup", "serve", "--platform", "gpu"),
            ("setup", "vllm"),
            ("serve", "--engine", "sglang", "--config", "configs/release/vlm_vllm_gpu.yaml"),
            ("serve", "--platform", "ppu", "--config", "configs/release/vlm_vllm_gpu.yaml"),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                plan(*arguments)

    def test_platform_is_read_from_runtime(self):
        with tempfile.TemporaryDirectory(prefix="ga-vllm-cli-") as directory:
            root = Path(directory)
            (root / "grounding_anything").mkdir()
            (root / "configs/release").mkdir(parents=True)
            shutil.copyfile(ROOT / "configs/release/vlm_vllm_ppu.yaml", root / "configs/release/vlm_vllm_ppu.yaml")
            with self.assertRaises(ValueError):
                plan("serve", "--engine", "vllm", root=root)
            (root / ".venv-vllm").mkdir()
            (root / ".venv-vllm/runtime.json").write_text('{"platform":"ppu"}')
            self.assertEqual(plan("serve", "--engine", "vllm", root=root)["command"][-1],
                             "configs/release/vlm_vllm_ppu.yaml")

    def test_release_and_eval_configs(self):
        import yaml
        for platform in ("gpu", "ppu"):
            config = yaml.safe_load((ROOT / f"configs/release/vlm_vllm_{platform}.yaml").read_text())
            self.assertEqual(config["entrypoint"], "vlm-vllm")
            self.assertEqual(config["outputs"], ["outputs/vllm-vlm"])
        old = yaml.safe_load((ROOT / "configs/eval/vlm.yaml").read_text())
        new = yaml.safe_load((ROOT / "configs/eval/vlm_vllm.yaml").read_text())
        self.assertEqual(new["mode"], "GAM")
        self.assertNotIn("decoder", new)
        self.assertEqual(new["run_id"], "vlm_vllm_smoke_001")
        new["run_id"] = old["run_id"]
        self.assertEqual(new, old)


class OverlayTests(unittest.TestCase):
    def test_canonical_transforms_are_idempotent_and_reject_unknown(self):
        for name, transform in PATCHERS.items():
            with self.subTest(name=name):
                source = (ROOT / "models/vlm" / name).read_text(encoding="utf-8")
                transformed = transform(source)
                self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(transformed)))
                self.assertEqual(transform(transformed), transformed)
                with self.assertRaises(ValueError):
                    transform(source + "\nUNRECOGNIZED_MODEL_CHANGE = True\n")

    def test_unprepared_model_transform_still_works(self):
        name = "modeling_groundinganything.py"
        source = (ROOT / "models/vlm" / name).read_text(encoding="utf-8")
        helper = ('try:\n    from transformers.utils.generic import is_flash_attention_requested\n'
                  'except ImportError:\n    def is_flash_attention_requested(config):\n'
                  '        return getattr(config, "_attn_implementation", None) == "flash_attention_2"')
        raw = source.replace(helper, "from transformers.utils.generic import is_flash_attention_requested", 1)
        raw = raw.replace("    _supports_attention_backend = True\n", "", 1)
        self.assertNotEqual(ast.dump(ast.parse(raw)), ast.dump(ast.parse(source)))
        self.assertEqual(ast.dump(ast.parse(PATCHERS[name](raw))), ast.dump(ast.parse(source)))

    def test_overlay_preserves_weights_and_checks_stale_adapter(self):
        with tempfile.TemporaryDirectory(prefix="ga-vllm-overlay-") as directory:
            root = Path(directory)
            model = make_checkpoint(root)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()}
            output = root / "outputs/vllm-vlm"
            create_overlay(model, output, root)
            manifest = verify_overlay(model, output, root)
            self.assertEqual(manifest["source_model"], "weights/vlm")
            self.assertEqual((output / "model.safetensors").resolve(), model / "model.safetensors")
            self.assertTrue((output / "model.safetensors").is_symlink())
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()})
            with self.assertRaises(FileExistsError):
                create_overlay(model, output, root)
            changed = output / "processing_groundinganything.py"
            changed.write_text(changed.read_text() + "\n# changed\n")
            with self.assertRaises(ValueError):
                verify_overlay(model, output, root)

    def test_dlm_and_missing_shards_fail_before_overlay_creation(self):
        with tempfile.TemporaryDirectory(prefix="ga-vllm-invalid-") as directory:
            root = Path(directory)
            model = make_checkpoint(root)
            config = json.loads((model / "config.json").read_text())
            config["architectures"] = ["GroundAnythingForConditionalGeneration"]
            (model / "config.json").write_text(json.dumps(config))
            output = root / "outputs/vllm-vlm"
            with self.assertRaisesRegex(ValueError, "DLM"):
                create_overlay(model, output, root)
            self.assertFalse(output.parent.exists())
            config["architectures"] = ["GroundAnythingVLMForConditionalGeneration"]
            (model / "config.json").write_text(json.dumps(config))
            (model / "model.safetensors.index.json").write_text('{"weight_map":{"x":"missing.safetensors"}}')
            with self.assertRaises(FileNotFoundError):
                create_overlay(model, output, root)
            (model / "model.safetensors.index.json").write_text('{"weight_map":{"x":"../escape.safetensors"}}')
            with self.assertRaises(ValueError):
                create_overlay(model, output, root)

    def test_dlm_tokenizer_and_overlapping_paths_fail(self):
        with tempfile.TemporaryDirectory(prefix="ga-vllm-invalid-") as directory:
            root = Path(directory)
            model = make_checkpoint(root)
            with self.assertRaises(ValueError):
                create_overlay(model, model / "overlay", root)
            (model / "tokenizer_config.json").write_text('{"added_tokens_decoder":{"0":{"content":"|<MASK>|"}}}')
            with self.assertRaisesRegex(ValueError, "DLM tokenizer"):
                source_files(model, root)


class RuntimeContractTests(unittest.TestCase):
    def test_server_command_limits(self):
        args = parser_for().parse_args(["--model", "weights/vlm", "--platform", "gpu"])
        command = build_command(args, Path("overlay"), "python-test")
        for key, value in (("--port", "8102"), ("--served-model-name", "groundinganything-vlm"),
                           ("--max-model-len", "16384"), ("--tensor-parallel-size", "1"),
                           ("--model-impl", "transformers"), ("--limit-mm-per-prompt", '{"image":1,"video":0}')):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--enforce-eager", command)
        args.gpu_memory_utilization = 1.0
        with self.assertRaises(ValueError):
            build_command(args, Path("overlay"))

    def test_environment_plan_preserves_accelerator_packages(self):
        with tempfile.TemporaryDirectory(prefix="ga-vllm-env-") as directory:
            root = Path(directory)
            target, commands = environment_plan(".venv-vllm", "gpu", root=root)
            self.assertFalse(target.exists())
            self.assertIn("--system-site-packages", commands[0])
            pip_commands = [c for c in commands if "pip" in c]
            self.assertTrue(all("--no-deps" in c for c in pip_commands))
            self.assertTrue(any(str(root / "requirements/vllm.txt") in c for c in commands))
            self.assertFalse(any("requirements/serve.txt" in " ".join(c) for c in commands))

    def test_http_client_preserves_both_gam_token_fields(self):
        from grounding_anything import GroundingAnything
        captured = []
        response = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        def fake_open(request, **kwargs):
            captured.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())
        with tempfile.TemporaryDirectory(prefix="ga-vllm-http-") as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"request-format-fixture")
            with patch("grounding_anything.client.urlopen", fake_open):
                GroundingAnything(base_url="http://127.0.0.1:8102/v1", model="groundinganything-vlm").predict(image, "cup")
        self.assertIs(captured[0]["skip_special_tokens"], False)
        self.assertIs(captured[0]["spaces_between_special_tokens"], False)
        self.assertEqual(captured[0]["model"], "groundinganything-vlm")


if __name__ == "__main__":
    unittest.main()
