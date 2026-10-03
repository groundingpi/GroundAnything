"""CPU integration tests with a local fake OpenAI-compatible HTTP service."""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "batch_predict.py"
SPEC = importlib.util.spec_from_file_location("batch_predict_example", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)

ANSWER = "<|object_ref_start|>car<|object_ref_end|><|box_start|><10><20><300><400><|box_end|>"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aG4sAAAAASUVORK5CYII=")


def completion(raw=ANSWER, finish="stop"):
    return {"choices": [{"message": {"content": raw}, "finish_reason": finish}],
            "usage": {"completion_tokens": 12}}


class BatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                cls.requests.append((self.path, json.loads(self.rfile.read(length))))
                status, payload = cls.responses.pop(0) if cls.responses else (200, completion())
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        type(self).requests, type(self).responses = [], []
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "image.png").write_bytes(PNG)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.row = {"id": "image-001", "image": "image.png", "task": "bbox",
                    "prompt": "Locate the target referred to by the following description: car."}
        self.write_rows([self.row])
        self.args = argparse.Namespace(input=str(self.input), output=str(self.output),
            base_url=f"http://127.0.0.1:{self.server.server_port}/v1", model="test-model",
            max_tokens=512, timeout=5.0, run_tag="checkpoint-a", api_key_env=None)

    def write_rows(self, rows):
        self.input.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def run_batch(self):
        with redirect_stdout(io.StringIO()):
            return batch.run(self.args)

    def results(self):
        return [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]

    def test_transport_and_successful_resume(self):
        self.assertEqual(self.run_batch(), {"success": 1, "failed": 0, "skipped": 0})
        route, payload = self.requests[0]
        self.assertEqual(route, "/v1/chat/completions")
        self.assertIs(payload["skip_special_tokens"], False)
        self.assertIs(payload["spaces_between_special_tokens"], False)
        image = payload["messages"][0]["content"][0]["image_url"]["url"]
        self.assertEqual(base64.b64decode(image.split(",", 1)[1]), PNG)
        record = self.results()[0]
        self.assertEqual(record["predictions"][0]["coordinates"], [[10, 20, 300, 400]])
        self.assertEqual(record["raw_output"], ANSWER)
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["usage"], {"completion_tokens": 12})
        self.assertEqual(self.run_batch(), {"success": 0, "failed": 0, "skipped": 1})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.results()), 1)

    def test_changed_input_or_config_does_not_resume_old_prediction(self):
        self.run_batch()
        self.row["prompt"] = "OCR task detect all the text in box format."
        self.write_rows([self.row])
        self.assertEqual(self.run_batch()["success"], 1)
        (self.root / "image.png").write_bytes(PNG + b"changed")
        self.assertEqual(self.run_batch()["success"], 1)
        self.args.model = "another-model"
        self.assertEqual(self.run_batch()["success"], 1)
        self.args.max_tokens = 1024
        self.assertEqual(self.run_batch()["success"], 1)
        self.args.run_tag = "checkpoint-b"
        self.assertEqual(self.run_batch()["success"], 1)
        self.assertEqual(len({r["request_id"] for r in self.results()}), 6)

    def test_failed_and_truncated_attempts_are_retained_and_retried(self):
        rows = [{**self.row, "id": str(i)} for i in range(3)]
        self.write_rows(rows)
        type(self).responses = [(200, completion(finish="length")),
                                (200, completion("not GAM")), (503, {"error": "busy"})]
        self.assertEqual(self.run_batch(), {"success": 0, "failed": 3, "skipped": 0})
        bad = self.results()
        self.assertEqual([r["status"] for r in bad], ["invalid_response", "invalid_response", "request_error"])
        self.assertEqual(bad[0]["raw_output"], ANSWER)
        self.assertTrue(all(not r["predictions"] for r in bad))
        self.assertEqual(self.run_batch(), {"success": 3, "failed": 0, "skipped": 0})
        self.assertEqual(len(self.results()), 6)

    def test_duplicate_manifest_ids_fail_before_requests(self):
        self.write_rows([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.run_batch()
        self.assertFalse(self.requests)
        self.assertFalse(self.output.exists())

    def test_corrupt_result_log_is_not_overwritten(self):
        damaged = '{"schema_version":1,"status":'
        self.output.write_text(damaged, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid result log"):
            self.run_batch()
        self.assertEqual(self.output.read_text(encoding="utf-8"), damaged)
        self.assertFalse(self.requests)

    def test_missing_image_is_logged_and_can_be_retried(self):
        self.row["image"] = "missing.png"
        self.write_rows([self.row])
        self.assertEqual(self.run_batch()["failed"], 1)
        self.assertFalse(self.requests)
        (self.root / "missing.png").write_bytes(PNG)
        self.assertEqual(self.run_batch()["success"], 1)
        self.assertEqual(len(self.results()), 2)

    def test_point_and_negative_formats(self):
        point = ANSWER.replace("<10><20><300><400>", "<10><20>")
        self.assertEqual(batch.parse_completion(completion(point), "point")["status"], "success")
        self.assertEqual(batch.parse_completion(completion(point), "bbox")["status"], "invalid_response")
        negative = ANSWER.replace("<10><20><300><400>", "None")
        result = batch.parse_completion(completion(negative), "bbox")
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["predictions"][0]["coordinates"])


if __name__ == "__main__":
    unittest.main()
