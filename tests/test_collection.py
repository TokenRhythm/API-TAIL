"""Network-free collection contract tests; all credentials and replies are fake."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error

from apitail.collect import collect
from apitail.cli import main


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts = self.root / "prompts.jsonl"
        self.prompts.write_text('\n'.join(json.dumps(x) for x in [
            {"pid": "first", "text": "Write one word."},
            {"pid": "second", "text": "Name a color."}]) + '\n', encoding="utf-8")
        self.config = self.root / "endpoint.json"
        self.output = self.root / "responses.jsonl"
        self.secret = "TEST_ONLY_CREDENTIAL_" + "VALUE"
        self.base = {"url": "https://provider.invalid/v1/chat/completions",
                     "model": "test-model", "api_key_env": "APITAIL_TEST_KEY",
                     "parameters": {"max_tokens": 8, "temperature": 0.7}}
        self.write_config()

    def write_config(self, **updates):
        self.config.write_text(json.dumps(dict(self.base, **updates)), encoding="utf-8")

    @contextlib.contextmanager
    def network(self, side_effect=None):
        # Block both urllib entrypoints so transport changes cannot make network calls.
        call = Mock(side_effect=side_effect)
        opener = Mock()
        opener.open = call
        with patch("urllib.request.urlopen", call), patch("urllib.request.build_opener", return_value=opener):
            yield call

    def reply(self, text="A color."):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": text},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 3}}).encode())

    def test_default_and_cli_are_network_free_without_keys(self):
        with self.network() as net, patch.dict(os.environ, {}, clear=True):
            plan = collect(self.config, self.prompts, self.output)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(["collect", "--config", str(self.config), "--prompts", str(self.prompts),
                      "--output", str(self.output)])
            self.assertFalse(plan["execute"])
            self.assertEqual(plan["requests"], 2)
            self.assertFalse(json.loads(stdout.getvalue())["execute"])
            net.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".jsonl.partial").exists())

    def test_success_payload_and_saved_responses(self):
        self.write_config(responses_per_prompt=2, system="Be concise.")
        with self.network(lambda *a, **kw: self.reply()) as net, patch.dict(os.environ, {"APITAIL_TEST_KEY": self.secret}):
            plan = collect(self.config, self.prompts, self.output, execute=True)
        self.assertEqual(plan["completed"], 4)
        self.assertEqual(net.call_count, 4)
        bodies = []
        for call in net.call_args_list:
            request = call.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer " + self.secret)
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "test-model")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["max_tokens"], 8)
            self.assertEqual(payload["temperature"], .7)
            self.assertEqual(payload["messages"][0], {"role": "system", "content": "Be concise."})
            self.assertEqual(payload["messages"][1]["role"], "user")
            bodies.append(payload)
        self.assertEqual(bodies[0], bodies[1])
        self.assertNotEqual(bodies[1]["messages"], bodies[2]["messages"])
        rows = [json.loads(x) for x in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["pid"], r["k"]) for r in rows], [("first", 0), ("first", 1), ("second", 0), ("second", 1)])
        self.assertTrue(all(r["text"] == "A color." and r["finish_reason"] == "stop" for r in rows))
        self.assertTrue(all(r["usage"]["completion_tokens"] == 3 for r in rows))
        self.assertNotIn(self.secret, self.output.read_text(encoding="utf-8") + json.dumps(plan))
        self.assertFalse(self.output.with_suffix(".jsonl.partial").exists())

    def test_error_redacts_provider_details_and_does_not_retry(self):
        errors = [urllib.error.HTTPError(self.base["url"], 429, self.secret, {}, None),
                  urllib.error.URLError(self.secret)]
        for error in errors:
            with self.subTest(type=type(error).__name__):
                target = self.root / (type(error).__name__ + ".jsonl")
                with self.network(error) as net, patch.dict(os.environ, {"APITAIL_TEST_KEY": self.secret}):
                    with self.assertRaises(RuntimeError) as caught:
                        collect(self.config, self.prompts, target, execute=True)
                    self.assertNotIn(self.secret, str(caught.exception))
                    self.assertEqual(net.call_count, 1)
                self.assertFalse(target.exists())
                self.assertNotIn(self.secret, target.with_suffix(".jsonl.partial").read_text(encoding="utf-8"))

    def test_partial_rows_survive_failure_without_duplicate_retry(self):
        with self.network([self.reply("First answer."), urllib.error.URLError(self.secret)]) as net, patch.dict(os.environ, {"APITAIL_TEST_KEY": self.secret}):
            with self.assertRaises(RuntimeError):
                collect(self.config, self.prompts, self.output, execute=True)
            self.assertEqual(net.call_count, 2)
        partial = self.output.with_suffix(".jsonl.partial")
        rows = [json.loads(x) for x in partial.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pid"], "first")
        self.assertNotIn(self.secret, partial.read_text(encoding="utf-8"))

    def test_embedded_credentials_rejected_before_network(self):
        cases = [{"api_key": self.secret}, {"authorization": self.secret},
                 {"headers": {"Authorization": self.secret}}, {"password": self.secret},
                 {"url": "https://user:" + self.secret + "@provider.invalid/v1"},
                 {"url": "https://provider.invalid/v1?api_key=" + self.secret},
                 {"parameters": {"headers": {"Authorization": self.secret}}},
                 {"parameters": {"password": self.secret}}]
        for update in cases:
            with self.subTest(fields=list(update)):
                self.write_config(**update)
                with self.network() as net:
                    with self.assertRaises(ValueError) as caught:
                        collect(self.config, self.prompts, self.output)
                    self.assertNotIn(self.secret, str(caught.exception))
                    net.assert_not_called()

    def test_extra_choices_cannot_silently_change_request_budget(self):
        self.write_config(parameters={"n": 2})
        with self.network() as net:
            with self.assertRaises(ValueError):
                collect(self.config, self.prompts, self.output)
            net.assert_not_called()

    def test_fractional_negative_or_boolean_request_counts_fail(self):
        for value in (0, -1, 1.5, True, "2"):
            with self.subTest(responses_per_prompt=value):
                self.write_config(responses_per_prompt=value)
                with self.network() as net:
                    with self.assertRaises(ValueError):
                        collect(self.config, self.prompts, self.output)
                    net.assert_not_called()

    def test_overwrite_and_partial_protected_before_network(self):
        for suffix in (".jsonl", ".jsonl.partial"):
            with self.subTest(suffix=suffix):
                retained = self.output.with_suffix(suffix)
                retained.write_text("retained-data", encoding="utf-8")
                with self.network() as net, patch.dict(os.environ, {"APITAIL_TEST_KEY": self.secret}):
                    with self.assertRaises(FileExistsError):
                        collect(self.config, self.prompts, self.output, execute=True)
                    net.assert_not_called()
                self.assertEqual(retained.read_text(encoding="utf-8"), "retained-data")
                retained.unlink()

    def test_malformed_environment_key_never_enters_transport_or_error(self):
        for ending in ("\n", "\r", "\t"):
            with self.subTest(control=repr(ending)):
                with self.network() as net, patch.dict(os.environ, {"APITAIL_TEST_KEY": self.secret + ending}):
                    with self.assertRaises(ValueError) as caught:
                        collect(self.config, self.prompts, self.output, execute=True)
                    self.assertNotIn(self.secret, str(caught.exception))
                    net.assert_not_called()

    def test_redirect_handler_does_not_forward_authorization(self):
        import apitail.collect as module
        handler = module.NoRedirect()
        request = __import__("urllib.request", fromlist=["Request"]).Request(
            self.base["url"], headers={"Authorization": "Bearer " + self.secret})
        self.assertIsNone(handler.redirect_request(request, None, 302, "redirect", {},
                                                 "https://different.invalid/"))


if __name__ == "__main__":
    unittest.main()
