"""Portable, non-pickle score banks and explicit prompt-level sampling."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{Path(path).name}:{lineno}: invalid JSON") from e
            if not isinstance(row, dict):
                raise ValueError(f"Line {lineno}: expected an object")
            if "_meta" not in row:
                rows.append(row)
    return rows


def prompt_map(path):
    result = {}
    for row in read_jsonl(path):
        pid = str(row["pid"])
        if pid in result:
            raise ValueError(f"Duplicate prompt id: {pid}")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Prompt {pid} has no text")
        result[pid] = row
    if not result:
        raise ValueError("No prompts")
    return result


def write_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class ScoreBank:
    S: np.ndarray
    V: np.ndarray
    grid: np.ndarray
    pid: np.ndarray
    k: np.ndarray
    n_tokens: np.ndarray
    metadata: dict
    extra: dict

    def __post_init__(self):
        for name, value in (("k", self.k), ("n_tokens", self.n_tokens)):
            raw = np.asarray(value)
            if raw.ndim != 1 or raw.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a one-dimensional integer array")
        if np.asarray(self.pid).ndim != 1:
            raise ValueError("pid must be one-dimensional")
        if not isinstance(self.metadata, dict) or not isinstance(self.extra, dict):
            raise ValueError("metadata and extra must be mappings")
        self.S = np.asarray(self.S, dtype=np.float64)
        self.V = np.asarray(self.V, dtype=np.float64)
        self.grid = np.asarray(self.grid, dtype=np.float64)
        self.pid = np.asarray(self.pid, dtype=str)
        self.k = np.asarray(self.k, dtype=np.int64)
        self.n_tokens = np.asarray(self.n_tokens, dtype=np.int64)
        n = len(self.pid)
        if np.any(self.k < 0) or np.any(self.pid == ""):
            raise ValueError("Response indices must be nonnegative and prompt ids nonempty")
        if self.S.shape != (n, len(self.grid)) or self.V.shape != self.S.shape:
            raise ValueError("S and V must have shape (responses, temperatures)")
        if n == 0 or self.grid.ndim != 1 or len(self.grid) == 0:
            raise ValueError("Empty score bank")
        if not np.all(np.isfinite(self.grid)) or np.any(self.grid <= 0) or np.any(np.diff(self.grid) <= 0):
            raise ValueError("Temperatures must be finite, positive and strictly increasing")
        if not np.all(np.isfinite(self.S)) or not np.all(np.isfinite(self.V)) or np.any(self.V < 0):
            raise ValueError("Scores must be finite and variances nonnegative")
        if self.k.shape != (n,) or self.n_tokens.shape != (n,) or np.any(self.n_tokens <= 0):
            raise ValueError("Every response must contain at least one scored token")
        if len(set(zip(self.pid.tolist(), self.k.tolist()))) != n:
            raise ValueError("Duplicate (pid, k) in score bank")
        for name, x in self.extra.items():
            if np.asarray(x).dtype.kind not in "biufUS" or np.asarray(x).ndim == 0:
                raise ValueError(f"Extra field {name} must be a non-object array")
            if np.asarray(x).shape[0] != n:
                raise ValueError(f"Extra field {name} has wrong response dimension")

    @property
    def prompt_ids(self):
        return sorted(set(self.pid.tolist()))

    def draws(self, budget, trials, rng, prompt_ids=None):
        if type(budget) is not int or budget <= 0 or type(trials) is not int or trials <= 0:
            raise ValueError("budget and trials must be positive integers")
        groups = {}
        for i, pid in enumerate(self.pid):
            groups.setdefault(str(pid), []).append(i)
        ids = self.prompt_ids if prompt_ids is None else list(prompt_ids)
        if len(set(ids)) != len(ids) or any(p not in groups for p in ids):
            raise ValueError("Prompt ids must be unique and present in this source")
        if budget > len(ids):
            raise ValueError(f"budget={budget} exceeds {len(ids)} distinct prompts")
        out = np.empty((trials, budget), dtype=np.int64)
        for i in range(trials):
            chosen = rng.choice(len(ids), budget, replace=False)
            out[i] = [rng.choice(groups[ids[j]]) for j in chosen]
        return out


def save_scores(path, bank, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(S=bank.S, V=bank.V, grid=bank.grid, pid=bank.pid, k=bank.k,
                   n_tokens=bank.n_tokens, metadata=np.asarray(json.dumps(bank.metadata, allow_nan=False)))
    for name, value in bank.extra.items():
        if name in payload:
            raise ValueError(f"Reserved score field {name}")
        payload[name] = np.asarray(value)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_scores(path):
    # Public artifacts must never require pickle execution.
    with np.load(path, allow_pickle=False) as a:
        required = {"S", "V", "grid", "pid", "k", "n_tokens", "metadata"}
        if not required <= set(a.files):
            raise ValueError(f"Missing score fields: {sorted(required - set(a.files))}")
        return ScoreBank(a["S"], a["V"], a["grid"], a["pid"], a["k"], a["n_tokens"],
                         json.loads(str(a["metadata"])), {n: a[n] for n in a.files if n not in required})


def compatible(banks):
    first = banks[0]
    for bank in banks[1:]:
        if not np.array_equal(first.grid, bank.grid):
            raise ValueError("Score banks use different temperature grids")
        # Compare score semantics, not file paths or endpoint names.
        for field in ("model_id", "model_config_sha256", "dtype", "scoring", "system", "chat_kwargs", "max_length"):
            if first.metadata.get(field) != bank.metadata.get(field):
                raise ValueError(f"Score banks disagree on {field}")
