# Test group: SHARED.
# Paper RQ: Shared input preparation for RQ1--RQ5.
# Checks: Portable input paths and output isolation; no data or model evaluation.
"""CPU-only path checks; no training, distance evaluation, or dataset loading."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "09_tools/data_preparation/archive"
SPEC = importlib.util.spec_from_file_location("preparation_paths", ARCHIVE / "paths.py")
PATHS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATHS)


class PreparationPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "paths.json"
        self.section = {}
        for stage, required in PATHS.STAGES.items():
            section = {"output_root": f"outputs/{stage}"}
            for key, kind in required.items():
                source = self.root / stage / key
                source.parent.mkdir(parents=True, exist_ok=True)
                if kind == "directory":
                    source.mkdir()
                else:
                    source.touch()
                section[key] = source.relative_to(self.root).as_posix()
            self.section[stage] = section
        self.write_config()

    def write_config(self):
        self.config.write_text(json.dumps(self.section), encoding="utf-8")

    def test_all_stage_paths_are_config_relative(self):
        for stage, required in PATHS.STAGES.items():
            with self.subTest(stage=stage):
                result = PATHS.resolve_paths(self.config, stage)
                self.assertEqual(result["output_root"], self.root / "outputs" / stage)
                for key in required:
                    self.assertEqual(result[key], self.root / stage / key)

    def test_check_only_creates_nothing(self):
        before = set(self.root.rglob("*"))
        for stage in PATHS.STAGES:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = PATHS.load_paths(stage, ["--config", str(self.config), "--check-only"])
            self.assertIsNone(result)
            self.assertEqual(json.loads(output.getvalue())["status"], "paths_verified")
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_missing_input_is_actionable(self):
        (self.root / "prepare_100k/porto_ids").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "porto_ids"):
            PATHS.resolve_paths(self.config, "prepare_100k")

    def test_missing_key_rejected(self):
        del self.section["prepare_100k"]["porto_ids"]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "porto_ids"):
            PATHS.resolve_paths(self.config, "prepare_100k")

    def test_absolute_input_supported(self):
        self.section["prepare_100k"]["porto_source"] = str(self.root / "prepare_100k/porto_source")
        self.write_config()
        self.assertTrue(PATHS.resolve_paths(self.config, "prepare_100k")["porto_source"].is_file())

    def test_output_must_not_overlap_source(self):
        for output in ("prepare_sdd/source_dir", "prepare_sdd/source_dir/generated", "."):
            with self.subTest(output=output):
                self.section["prepare_sdd"]["output_root"] = output
                self.write_config()
                with self.assertRaisesRegex(ValueError, "overlaps input"):
                    PATHS.resolve_paths(self.config, "prepare_sdd")

    def test_nonempty_output_protected(self):
        output = self.root / "outputs/prepare_100k"
        output.mkdir(parents=True)
        (output / "existing.json").write_text("{}")
        with self.assertRaisesRegex(FileExistsError, "new or empty"):
            PATHS.resolve_paths(self.config, "prepare_100k")

    def test_empty_output_allowed(self):
        output = self.root / "outputs/prepare_100k"
        output.mkdir(parents=True)
        self.assertEqual(PATHS.resolve_paths(self.config, "prepare_100k")["output_root"], output)

    def test_all_launchers_check_from_other_directory(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for stage in PATHS.STAGES:
            with self.subTest(stage=stage):
                process = subprocess.run(
                    [sys.executable, str(ARCHIVE / (stage + ".py")),
                     "--config", str(self.config), "--check-only"],
                    cwd=self.root, env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout)["stage"], stage)
        self.assertFalse((self.root / "outputs").exists())

    def test_imports_create_no_files(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import prepare_100k, prepare_sdd, audit_cities, prepare_city_training")
        process = subprocess.run([sys.executable, "-B", "-c", code, str(ARCHIVE)],
                                 cwd=self.root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "")
        self.assertFalse((self.root / "outputs").exists())

    def test_utf8_bom_config_supported(self):
        self.config.write_text(json.dumps(self.section), encoding="utf-8-sig")
        self.assertEqual(PATHS.resolve_paths(self.config, "prepare_sdd")["output_root"],
                         self.root / "outputs/prepare_sdd")


if __name__ == "__main__":
    unittest.main()
