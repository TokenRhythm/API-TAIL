"""UTF-8 and CLI contracts. GBK defaults are simulated, not a Windows run."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

import numpy as np
from apitail import __version__
from apitail.cli import main
from apitail.collect import collect
from apitail.generate import generate
from apitail.hf import load_model
from apitail.io import ScoreBank, save_scores, write_json
from apitail.pipeline import calibrate, audit, evaluate

TORCH = importlib.util.find_spec("torch") is not None
TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@contextlib.contextmanager
def gbk_path_defaults(root):
    """Only emulate omitted Path text encodings for this test's own files."""
    original_read, original_write = Path.read_text, Path.write_text
    implicit = []

    def inside(path):
        try:
            Path(path).relative_to(root)
            return True
        except ValueError:
            return False

    def read(path, encoding=None, errors=None, **kwargs):
        if inside(path) and encoding is None:
            implicit.append((str(path), "read"))
            encoding = "gbk"
        return original_read(path, encoding=encoding, errors=errors, **kwargs)

    def write(path, data, encoding=None, errors=None, **kwargs):
        if inside(path) and encoding is None:
            implicit.append((str(path), "write"))
            encoding = "gbk"
        return original_write(path, data, encoding=encoding, errors=errors, **kwargs)

    with patch.object(Path, "read_text", read), patch.object(Path, "write_text", write):
        yield implicit


class TextAndCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="apitail-utf8-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "中文目录"
        self.root.mkdir()
        self.prompts = self.root / "提问.jsonl"
        self.prompts.write_text(json.dumps({"pid": "问题甲", "text": "请说一种颜色。"}, ensure_ascii=False) + '\n', encoding="utf-8")
        self.endpoint = self.root / "采集配置.json"
        write_json(self.endpoint, {"url": "https://provider.invalid/v1/chat/completions",
                   "model": "测试模型", "api_key_env": "APITAIL_TEXT_TEST_KEY",
                   "system": "只用中文回答。", "parameters": {"max_tokens": 8}})

    def bank(self, prefix, values):
        n = len(values)
        return ScoreBank(np.asarray(values, dtype=float)[:, None], np.ones((n, 1)),
                         np.asarray([1.]), np.asarray([prefix + str(i) for i in range(n)]),
                         np.zeros(n, dtype=np.int64), np.full(n, 4, dtype=np.int64),
                         {"model_id": "中文参考模型", "model_config_sha256": "a" * 64,
                          "dtype": "float32", "scoring": "fixture", "system": "请简答。",
                          "chat_kwargs": {}, "max_length": 64}, {})

    def test_chinese_calibration_audit_and_evaluation_under_gbk_default(self):
        save_scores(self.root / "默认.npz", self.bank("校准", [0, 0, 0, 0]))
        save_scores(self.root / "高尾部.npz", self.bank("校准", [4, 0, 0, 0]))
        save_scores(self.root / "审计.npz", self.bank("审计", [4, 4]))
        config, rule = self.root / "\u6821\u51c6\u914d\u7f6e.json", self.root / "\u9608\u503c.json"
        write_json(config, {"members": {"默认": "默认.npz", "高尾部": "高尾部.npz"},
                            "budget": 2, "trials": 200, "method": "fixed", "seed": 19})
        with gbk_path_defaults(self.root) as implicit:
            calibration = calibrate(config, rule)
            result = audit(self.root / "审计.npz", rule, self.root / "判定.json")
            evaluation = evaluate(self.root / "审计.npz", rule, self.root / "评估.json", trials=20)
            self.assertEqual(implicit, [])
        self.assertEqual(calibration["decisive"], "高尾部")
        self.assertEqual(calibration["threshold"], 1)
        self.assertEqual(calibration["score_semantics"]["system"], "请简答。")
        self.assertTrue(result["reject"])
        self.assertEqual(evaluation["rate"], 1.)
        decoded = json.loads(rule.read_bytes().decode("utf-8"))
        self.assertEqual(decoded["calibration_prompt_ids"], ["校准0", "校准1", "校准2", "校准3"])

    def test_chinese_collection_config_payload_and_response_under_gbk_default(self):
        answer = {"choices": [{"message": {"content": "蓝色。"}, "finish_reason": "stop"}]}
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps(answer, ensure_ascii=False).encode("utf-8"))
        output = self.root / "\u91c7\u96c6\u7ed3\u679c.jsonl"
        with gbk_path_defaults(self.root) as implicit, patch("urllib.request.build_opener", return_value=opener), patch.dict(os.environ, {"APITAIL_TEXT_TEST_KEY": "dummy-credential"}):
            plan = collect(self.endpoint, self.prompts, output)
            opener.open.assert_not_called()
            result = collect(self.endpoint, self.prompts, output, execute=True)
            self.assertEqual(implicit, [])
        self.assertEqual(plan["model"], "测试模型")
        self.assertEqual(result["completed"], 1)
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "测试模型")
        self.assertEqual(payload["messages"], [{"role": "system", "content": "只用中文回答。"},
                                              {"role": "user", "content": "\u8bf7\u8bf4\u4e00\u79cd\u989c\u8272\u3002"}])
        row = json.loads(output.read_bytes().decode("utf-8"))
        self.assertEqual((row["pid"], row["text"]), ("问题甲", "蓝色。"))

    @unittest.skipUnless(TORCH and TRANSFORMERS, "Optional torch/transformers extras are not installed")
    def test_chinese_generation_config_and_output_under_gbk_default(self):
        import torch
        config, output = self.root / "生成配置.json", self.root / "生成结果.jsonl"
        write_json(config, {"system": "只用中文回答。", "seed": 2,
                            "sampling": {"temperature": .7, "max_tokens": 2}})
        model = Mock()
        model.generation_config = SimpleNamespace(eos_token_id=3)
        model.generate.return_value = torch.tensor([[0, 1, 2, 3]])
        tokenizer = SimpleNamespace(pad_token_id=None, decode=lambda *a, **k: "蓝色。")
        with gbk_path_defaults(self.root) as implicit, patch("apitail.generate.load_model", return_value=(torch, tokenizer, model)), patch("apitail.generate.chat_ids", return_value=[0, 1]) as chat:
            result = generate("mock-model", self.prompts, config, output, device="cpu", local_files_only=True)
            self.assertEqual(implicit, [])
        self.assertEqual(chat.call_args.args[1:3], ("请说一种颜色。", "只用中文回答。"))
        rows = [json.loads(line) for line in output.read_bytes().decode("utf-8").splitlines()]
        self.assertEqual(rows[0]["_meta"]["config"]["system"], "\u53ea\u7528\u4e2d\u6587\u56de\u7b54\u3002")
        self.assertEqual((rows[1]["pid"], rows[1]["text"]), ("问题甲", "蓝色。"))
        self.assertEqual(result["responses"], 1)

    def test_cli_provider_failures_have_no_traceback_or_credentials(self):
        sentinel = "TEST_ONLY_CREDENTIAL_" + "VALUE"
        errors = [urllib.error.HTTPError("https://provider.invalid", 429, sentinel, {}, None),
                  urllib.error.URLError(sentinel)]
        for i, error in enumerate(errors):
            with self.subTest(error=type(error).__name__):
                opener = Mock()
                opener.open.side_effect = error
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch("urllib.request.build_opener", return_value=opener), patch.dict(os.environ, {"APITAIL_TEXT_TEST_KEY": sentinel}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        main(["collect", "--config", str(self.endpoint), "--prompts", str(self.prompts),
                              "--output", str(self.root / f"failure-{i}.jsonl"), "--execute"])
                self.assertEqual(caught.exception.code, 2)
                self.assertTrue(stderr.getvalue().startswith("apitail: Provider"))
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertNotIn(sentinel, stdout.getvalue() + stderr.getvalue())
                self.assertEqual(opener.open.call_count, 1)

    def test_cli_runtime_and_dependency_errors_are_concise(self):
        for error in (RuntimeError("CUDA unavailable"), ImportError("No module named 'torch'"), OSError("Read failed")):
            with self.subTest(error=type(error).__name__):
                stderr = io.StringIO()
                with patch("apitail.hf.score", side_effect=error), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        main(["score", "--model", "mock-model", "--prompts", "p.jsonl",
                              "--responses", "r.jsonl", "--output", "out.npz"])
                self.assertEqual(caught.exception.code, 2)
                self.assertEqual(stderr.getvalue(), "apitail: " + str(error) + "\n")
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_version_uses_package_version(self):
        stdout = io.StringIO()
        test_version = __version__ + ".test"
        with patch("apitail.cli.__version__", test_version), contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as caught:
                main(["--version"])
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stdout.getvalue().strip(), f"apitail {test_version}")

    def test_model_loader_uses_transformers5_dtype_keyword(self):
        fake_torch = SimpleNamespace(bfloat16=object(), float32=object(), float16=object())
        tokenizer, model = Mock(), Mock()
        fake_transformers = SimpleNamespace(AutoTokenizer=Mock(), AutoModelForCausalLM=Mock())
        fake_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        fake_transformers.AutoModelForCausalLM.from_pretrained.return_value = model
        with patch("apitail.hf.deterministic_torch", return_value=fake_torch), patch.dict(sys.modules, {"transformers": fake_transformers}):
            actual = load_model("mock-model", device="cpu", dtype="float32", local_files_only=True)
        fake_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "mock-model", dtype=fake_torch.float32, device_map="cpu", local_files_only=True)
        model.eval.assert_called_once_with()
        self.assertEqual(actual, (fake_torch, tokenizer, model))


if __name__ == "__main__":
    unittest.main()
