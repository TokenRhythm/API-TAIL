"""Boundary checks for the release scanner using synthetic path strings."""
import json
from pathlib import Path
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCAN = runpy.run_path(str(ROOT / "scripts" / "check_release.py"))["check"]
FORWARD = chr(47)
BACK = chr(92)


def unix_path(group="home"):
    return FORWARD + FORWARD.join((group, "example researcher", "project", "file.py"))


def windows_path(separator=BACK, profile="Users"):
    return "C:" + separator + separator.join((profile, "Example Researcher", "project", "file.py"))


class ReleaseScannerTests(unittest.TestCase):
    def scan_text(self, value):
        with tempfile.TemporaryDirectory(prefix="apitail-scan-") as folder:
            root = Path(folder)
            # Keep mandatory-file validation separate from path classification.
            files = {"README.md": "Scanner fixture.\n", "LICENSE": "Fixture.\n",
                     "pyproject.toml": "# Fixture\n", "src/apitail/core.py": "pass\n",
                     "data/paper/manifest.json": "{}\n", "case.txt": value}
            for name, text in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return SCAN(root)

    def test_relative_data_links_and_nested_paths_are_not_private(self):
        paths = [".." + FORWARD + FORWARD.join(("data", "paper", "manifest.json")),
                 "cache" + unix_path(), "." + unix_path("data")]
        result = self.scan_text("\n".join(paths))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["issues"], [])

    def test_urls_with_data_or_profile_segments_are_not_private(self):
        urls = ["https://example.org" + FORWARD + FORWARD.join(("data", "public", "file.json")),
                "https://example.org" + unix_path("Users"),
                "https://example.org" + FORWARD + windows_path(FORWARD)]
        result = self.scan_text("\n".join(urls))
        self.assertEqual(result["issues"], [])

    def test_absolute_unix_profiles_are_flagged(self):
        for group in ("home", "Users", "data"):
            with self.subTest(group=group):
                result = self.scan_text(unix_path(group))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["issues"], [{"file": "case.txt", "issue": "private_path", "line": 1}])

    def test_absolute_windows_profiles_and_spaces_are_flagged(self):
        for separator, profile in ((BACK, "Users"), (FORWARD, "Users"), (BACK, "Documents and Settings")):
            with self.subTest(separator=repr(separator), profile=profile):
                result = self.scan_text(windows_path(separator, profile))
                self.assertEqual(result["issues"], [{"file": "case.txt", "issue": "private_path", "line": 1}])

    def test_json_escaped_windows_path_is_flagged(self):
        result = self.scan_text(json.dumps({"path": windows_path()}))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["issues"], [{"file": "case.txt", "issue": "private_path", "line": 1}])

    def test_quoted_inline_path_reports_location_without_echoing_path(self):
        value = unix_path("data")
        result = self.scan_text('first line\nsource = "' + value + '"\n')
        self.assertEqual(result["issues"], [{"file": "case.txt", "issue": "private_path", "line": 2}])
        self.assertNotIn(value, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
