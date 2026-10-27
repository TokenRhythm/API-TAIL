"""OpenAI-compatible text collection. Dry-run by default; keys are environment-only."""
import json
import os
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from .io import prompt_map
from .validation import integer, number, reject_credentials, CREDENTIAL_KEYS


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer credential to a redirected endpoint.
        return None


def collect(config_path, prompts, output, execute=False, limit=None):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Endpoint config must be an object")
    reject_credentials(cfg)
    url = cfg["url"]
    parsed = urllib.parse.urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain credentials")
    if any(key.lower() in CREDENTIAL_KEYS for key, value in urllib.parse.parse_qsl(parsed.query)):
        raise ValueError("URL query must not contain credentials")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
        raise ValueError("Use HTTPS, or a local HTTP test endpoint")
    P = prompt_map(prompts)
    n = integer(cfg.get("responses_per_prompt", 1), "responses_per_prompt", 1)
    items = list(P.items())
    if limit is not None:
        items = items[:integer(limit, "limit", 1)]
    params = cfg.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("parameters must be an object")
    if any(k in params for k in ("messages", "model", "stream", "n")):
        raise ValueError("Reserved request parameter; one completion is requested per call")
    timeout = number(cfg.get("timeout", 120), "timeout", minimum=0.001)
    plan = dict(model=cfg["model"], prompts=len(items), requests=len(items)*n,
                execute=bool(execute), output=str(output),
                warning="Executing sends prompt text to the configured provider and may incur charges.")
    if not execute:
        return plan
    key = os.environ.get(cfg.get("api_key_env", "OPENROUTER_API_KEY"))
    if not key:
        raise ValueError("API key environment variable is not set")
    if not key.isascii() or any(ord(c) <= 32 or ord(c) == 127 for c in key):
        raise ValueError("API key contains invalid whitespace or control characters")
    dst = Path(output)
    if dst.exists() or dst.with_suffix(dst.suffix + ".partial").exists():
        raise FileExistsError("Output or partial output exists; retain it and choose a new path")
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_suffix(dst.suffix + ".partial")
    completed = 0
    opener = urllib.request.build_opener(NoRedirect())
    with open(partial, "x", encoding="utf-8") as out:
        for pid, prompt in items:
            messages = ([{"role": "system", "content": cfg["system"]}] if cfg.get("system") else [])
            messages.append({"role": "user", "content": prompt["text"]})
            for k in range(n):
                body = dict(params, model=cfg["model"], messages=messages, stream=False)
                request = urllib.request.Request(url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
                # Ambiguous failures are not retried: they may already be billable.
                try:
                    with opener.open(request, timeout=timeout) as response:
                        result = json.load(response)
                except urllib.error.HTTPError as e:
                    raise RuntimeError(f"Provider returned HTTP {e.code}; retained {completed} completed responses") from None
                except urllib.error.URLError:
                    raise RuntimeError(f"Provider request failed; retained {completed} completed responses") from None
                choices = result.get("choices", [])
                if len(choices) != 1:
                    raise ValueError("Expected exactly one returned completion")
                choice = choices[0]
                text = choice.get("message", {}).get("content")
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Provider returned no text for {(pid, k)}")
                row = dict(pid=pid, k=k, text=text, finish_reason=choice.get("finish_reason"),
                           usage=result.get("usage", {}))
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
    os.replace(partial, dst)
    plan["completed"] = completed
    return plan
