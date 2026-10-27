"""Copy JSON while rebasing path strings beginning with --old-root.

Does not copy assets or convert endpoint IDs. Existing outputs are not overwritten."""
import argparse
import json
from pathlib import Path


def rebase(value, old_root, new_root):
    if isinstance(value, str):
        old = old_root.rstrip("/")
        if value == old or value.startswith(old + "/"):
            return str(new_root) + value[len(old):]
        return value
    if isinstance(value, list):
        return [rebase(x, old_root, new_root) for x in value]
    if isinstance(value, dict):
        return {k: rebase(v, old_root, new_root) for k, v in value.items()}
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--archive-root", required=True)
    p.add_argument("--old-root", required=True, help="Original absolute project root recorded in this JSON")
    a = p.parse_args()
    root = Path(a.archive_root).expanduser().resolve()
    if not root.is_dir():
        p.error("--archive-root must exist")
    data = json.loads(Path(a.input).read_text(encoding="utf-8"))
    with Path(a.output).open("x", encoding="utf-8") as f:
        json.dump(rebase(data, a.old_root, root), f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
