"""Read-only source, artifact and link checks."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys

IGNORE = {".git", "__pycache__", "build", "dist", "work", ".venv", "venv"}
TEXT = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".cff"}
def check(root):
    issues, count, size = [], 0, 0
    patterns = [
        ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
        ("api_key", re.compile(r"(?:sk-|hf_)[A-Za-z0-9_-]{24,}")),
        ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("credential_url", re.compile(r"https?://[^/\s:@]+:[^/\s@]+@")),
        ("private_path", re.compile(
            r"(?<![\w./\\:])(?:/(?:home|Users|data)/[^/\r\n\"'<>]+/|"
            r"[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\r\n\"'<>]+[\\/]+)")),
    ]
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if any(x in IGNORE or x.endswith(".egg-info") for x in rel.parts) or not f.is_file():
            continue
        count += 1; size += f.stat().st_size
        if f.is_symlink():
            issues.append({"file": str(rel), "issue": "symlink must not enter release"})
        if f.stat().st_size >= 50 * 1024**2:
            issues.append({"file": str(rel), "issue": "file exceeds 50 MiB"})
        if f.suffix.lower() in {".pkl", ".pt", ".pth", ".safetensors"} or f.name.startswith(".env"):
            issues.append({"file": str(rel), "issue": "private/model/unsafe artifact"})
        if f.suffix in TEXT or f.name in {"LICENSE", "NOTICE", ".gitignore"}:
            s = f.read_text(encoding="utf-8")
            if f.suffix == ".py":
                try: ast.parse(s, filename=str(rel))
                except SyntaxError as e: issues.append({"file": str(rel), "issue": f"syntax error at line {e.lineno}"})
            for label, pattern in patterns:
                for match in pattern.finditer(s):
                    issues.append({"file": str(rel), "issue": label, "line": s[:match.start()].count("\n")+1})
            if f.suffix == ".md":
                for link in re.findall(r"\]\(([^)]+)\)", s):
                    if "://" in link or link.startswith(("#", "mailto:")):
                        continue
                    target = link.split("#")[0].split(" ")[0]
                    if target and not (f.parent/target).exists():
                        issues.append({"file": str(rel), "issue": "missing document link", "target": target})
    for name in ("README.md", "LICENSE", "pyproject.toml", "src/apitail/core.py", "data/paper/manifest.json"):
        if not (root/name).is_file():
            issues.append({"file": name, "issue": "missing required release file"})
    return {"status": "passed" if not issues else "failed", "files_checked": count,
            "bytes_checked": size, "issues": issues,
            "scope": "Static heuristic checks; not a guarantee that every possible secret or copyright issue is absent."}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = check(args.root.resolve())
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"release_check.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
if __name__ == "__main__": main()
