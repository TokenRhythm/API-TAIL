"""Run a source-archive command with source-relative imports.

Evaluation requires compatible configurations, score caches and model assets."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ENTRIES = {
    "score": "jobs/paper2/hf_score.py",
    "generate": "jobs/paper2/gen_hf.py",
    "generate-vllm": "bbp/gen_multi.py",
    "evaluate": "jobs/paper2/paper2_eval.py",
    "met-native": "jobs/paper2/metnat_eval2.py",
    "met-reference": "bbp/cmp_met_native.py",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", help="Location of separately supplied original assets")
    p.add_argument("entry", choices=ENTRIES)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.archive_root:
        root = Path(a.archive_root).expanduser().resolve()
        if not root.is_dir():
            p.error("--archive-root must be an existing directory")
        os.environ["APITAIL_ARCHIVE_ROOT"] = str(root)
    args = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
    script = Path(__file__).resolve().parent / "code" / ENTRIES[a.entry]
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
