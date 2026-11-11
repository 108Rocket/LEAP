# Test group: SHARED.
# Paper RQ: Shared data conversion for RQ1--RQ5.
# Checks: File formats and provenance only; no model or accuracy evaluation.
"""CPU-only conversion regressions using small synthetic trajectories."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / '09_tools/data_preparation/archive'
sys.path.insert(0, str(ARCHIVE))
from prepare_100k import save_paths

SPEC = importlib.util.spec_from_file_location('pack_data', ROOT / '09_tools/pack_data.py')
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class PackData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.npz'
        self.output = self.root / 'packed.npz'
        self.rows = [np.array([[0.25, -0.5], [1.0, 2.0]], np.float64),
                     np.array([[3.0, 4.0], [5.0, 6.0], [7.5, 8.5]], np.float64)]
        self.points = np.concatenate(self.rows)
        self.ids = np.array([710, 9], np.int64)
        self.args = ['--input', str(self.source), '--output', str(self.output),
                     '--checkpoint', '100k.porto.hausdorff']

    def convert(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            PACK.main(self.args)
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(json.loads(output.getvalue())['geometry_transformation'], 'none')
        with np.load(self.output, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def assert_geometry(self, result, points=None):
        expected = self.points if points is None else points
        self.assertEqual(result['points'].dtype, expected.dtype)
        self.assertEqual(result['points'].tobytes(), expected.tobytes())
        np.testing.assert_array_equal(result['offsets'], [0, 2, 5])
        self.assertEqual(result['offsets'].dtype, np.dtype('int64'))
        entry = PACK.resolve('100k.porto.hausdorff')
        asset = json.loads(entry['asset_path'].read_text(encoding='utf-8'))
        for key in ('geometry_mean', 'geometry_std'):
            np.testing.assert_array_equal(result[key], asset[key])

    def reject(self, message):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                PACK.main(self.args)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, output.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(before, hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_historical_writer_to_converter(self):
        save_paths(self.source, self.rows, self.ids)
        result = self.convert()
        self.assert_geometry(result)
        np.testing.assert_array_equal(result['source_ids'], self.ids)
        self.assertEqual(result['source_ids'].dtype, self.ids.dtype)

    def test_real_cli_from_other_directory(self):
        save_paths(self.source, self.rows, self.ids)
        process = subprocess.run([sys.executable, '-B', str(ROOT / '09_tools/pack_data.py'),
                                  *self.args], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)['trajectories'], 2)
        with np.load(self.output, allow_pickle=False) as result:
            self.assert_geometry(result)
            np.testing.assert_array_equal(result['source_ids'], self.ids)

    def test_offsets_input_and_float32_preserved(self):
        points = self.points.astype(np.float32)
        np.savez(self.source, points=points, offsets=[0, 2, 5], source_ids=self.ids)
        result = self.convert()
        self.assert_geometry(result, points)
        np.testing.assert_array_equal(result['source_ids'], self.ids)

    def test_lengths_without_source_ids(self):
        np.savez(self.source, points=self.points, lengths=[2, 3])
        result = self.convert()
        self.assert_geometry(result)
        self.assertNotIn('source_ids', result)

    def test_offsets_without_source_ids(self):
        np.savez(self.source, points=self.points, offsets=[0, 2, 5])
        result = self.convert()
        self.assert_geometry(result)
        self.assertNotIn('source_ids', result)

    def test_matching_lengths_and_offsets(self):
        np.savez(self.source, points=self.points, lengths=[2, 3], offsets=[0, 2, 5])
        self.assert_geometry(self.convert())

    def test_inconsistent_lengths_and_offsets(self):
        np.savez(self.source, points=self.points, lengths=[2, 3], offsets=[0, 3, 5])
        self.reject('different trajectories')

    def test_invalid_lengths(self):
        for lengths in ([2, 4], [1, 4], [0, 5], [-2, 7], [2.0, 3.0], [[2, 3]], [],
                        np.array([2**63, 2**63 + 5], dtype=np.uint64)):
            with self.subTest(lengths=lengths):
                np.savez(self.source, points=self.points, lengths=lengths)
                self.reject('lengths must be integers')

    def test_invalid_offsets(self):
        for offsets in ([0, 1, 5], [0, 2, 4], [0, -2, 5], [1, 2, 5],
                        [0.0, 2.0, 5.0], [[0, 2, 5]],
                        np.array([0, 2**64 - 1, 5], dtype=np.uint64)):
            with self.subTest(offsets=offsets):
                np.savez(self.source, points=self.points, offsets=offsets)
                self.reject('Invalid finite packed geometry')

    def test_invalid_source_ids(self):
        for ids in ([710], [[710, 9]], [710.0, 9.0]):
            with self.subTest(ids=ids):
                np.savez(self.source, points=self.points, lengths=[2, 3], source_ids=ids)
                self.reject('one integer source ID per trajectory')

    def test_missing_fields(self):
        for fields in ({'points': self.points}, {'offsets': [0, 2, 5]}):
            with self.subTest(fields=list(fields)):
                np.savez(self.source, **fields)
                self.reject('NPZ requires points')

    def test_nonfinite_coordinates(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                points = self.points.copy()
                points[0, 0] = value
                np.savez(self.source, points=points, lengths=[2, 3])
                self.reject('Invalid finite packed geometry')

    def test_json_still_supported(self):
        self.source = self.root / 'source.json'
        self.args[1] = str(self.source)
        self.source.write_text(json.dumps([row.tolist() for row in self.rows]), encoding='utf-8')
        self.assert_geometry(self.convert())

    def test_existing_output_protected(self):
        save_paths(self.source, self.rows, self.ids)
        self.output.write_bytes(b'keep existing output')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                PACK.main(self.args)
        self.assertEqual(self.output.read_bytes(), b'keep existing output')

    def test_pandas_declared_in_both_install_paths(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest('Dependency metadata check requires Python 3.11+')
        metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertIn('pandas>=2.2,<3', metadata['project']['dependencies'])
        self.assertIn('pandas>=2.2,<3',
                      (ROOT / '10_requirements/reproduction.txt').read_text().splitlines())


if __name__ == '__main__':
    unittest.main()
