# Test group: RQ1.
# Paper RQ: RQ1: Overall Performance.
# Checks: Baseline source provenance and native interfaces; no baseline inference.
# Role: supporting regression tests. See 08_tests/tests/README.md for the paper mapping.

"""Author-source boundaries, without importing any external model."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT / '07_external_models/external_models'


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, EXTERNAL / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSTALL = load('leap_author_sources_test', 'install_sources.py')
with patch.dict(sys.modules, install_sources=INSTALL):
    RUN = load('leap_author_runner_test', 'run.py')


class AuthorInterfaceTests(unittest.TestCase):
    def test_source_registry_has_eight_primary_and_three_optional_models(self):
        sources = INSTALL.specifications()
        registry = json.loads((EXTERNAL / 'registry.json').read_text())
        self.assertEqual(set(sources), set(registry))
        self.assertEqual(len(sources), 11)
        self.assertEqual(sum(not row['optional'] for row in sources.values()), 8)
        for row in sources.values():
            self.assertEqual(set(row), {'repository', 'commit', 'directory', 'bundled', 'optional'})
            self.assertTrue((EXTERNAL / row['directory'] / 'README.md').is_file())

    def test_source_verification_rejects_parent_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'TrajCL'
            directory.mkdir()
            with patch.object(INSTALL, 'git', return_value=folder):
                with self.assertRaisesRegex(RuntimeError, 'separate author checkout'):
                    INSTALL.verify_source('TrajCL', folder)

    def check_verification(self, head=None, dirty='', extras=''):
        with tempfile.TemporaryDirectory() as folder:
            directory = (Path(folder) / 'TrajCL').resolve()
            directory.mkdir()
            commit = INSTALL.specifications()['TrajCL']['commit']
            replies = [str(directory), head or commit, dirty, extras]
            with patch.object(INSTALL, 'git', side_effect=replies):
                return INSTALL.verify_source('TrajCL', folder)

    def test_clean_source_verifies(self):
        self.check_verification()

    def test_wrong_commit_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'commit differs'):
            self.check_verification(head='0' * 40)

    def test_modified_author_source_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'tracked changes'):
            self.check_verification(dirty=' M model/attention.py')

    def test_injected_source_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'untracked source'):
            self.check_verification(extras='utils/replacement.py')

    def test_non_source_outputs_allowed(self):
        self.check_verification(extras='runs/metrics.json\ncheckpoints/best.pt')

    def test_install_never_rewrites_existing_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'TrajCL'
            destination.mkdir()
            with patch.object(INSTALL, 'verify_source', return_value=destination), \
                    patch.object(INSTALL, 'git') as git:
                self.assertEqual(INSTALL.main(['--model', 'TrajCL', '--source-root', folder]), 0)
                git.assert_not_called()

    def test_missing_source_verification_does_not_clone(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(INSTALL, 'git') as git:
                with self.assertRaises(FileNotFoundError):
                    INSTALL.main(['--model', 'TrajCL', '--source-root', folder, '--verify-only'])
                git.assert_not_called()

    def test_install_uses_only_clone_and_detached_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder).resolve() / 'TrajCL'
            with patch.object(INSTALL, 'verify_source', return_value=destination), \
                    patch.object(INSTALL, 'git') as git:
                INSTALL.main(['--model', 'TrajCL', '--source-root', folder])
                self.assertEqual([call.args[0] for call in git.call_args_list], ['clone', 'checkout'])
                self.assertEqual(git.call_args_list[1].args[1], '--detach')

    def test_lb_fails_before_loading_dependencies_or_sources(self):
        with patch.object(RUN, 'verify_source') as verify:
            with self.assertRaises(SystemExit) as raised:
                RUN.main(['LB-TrajRep'])
            self.assertEqual(raised.exception.code, 2)
            verify.assert_not_called()
        self.assertEqual(ast.parse((EXTERNAL / 'bounds/LB-TrajRep/interface.py').read_text()).body, [])

    def test_native_arguments_and_exit_code_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'main.py').write_text('pass\n')
            with patch.object(RUN, 'verify_source', return_value=root), \
                    patch.object(RUN.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as run:
                result = RUN.main(['TrajCL', '--entrypoint', 'main.py', '--', '--dataset', 'porto'])
                self.assertEqual(result, 7)
                run.assert_called_once_with(
                    [sys.executable, str(root / 'main.py'), '--dataset', 'porto'], cwd=root,
                )

    def test_runner_rejects_entrypoint_outside_author_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(RUN, 'verify_source', return_value=root), \
                    patch.object(RUN.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    RUN.main(['TrajCL', '--entrypoint', '../other.py'])
                run.assert_not_called()

    def test_no_model_replacement_or_custom_training_is_shipped(self):
        forbidden = {'MethodType', 'backward', 'load_state_dict_from_url'}
        for path in EXTERNAL.rglob('*.py'):
            if 'upstream' in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.fail('Model class in interface layer: ' + str(path))
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, forbidden, str(path))
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, forbidden, str(path))
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        self.assertFalse(isinstance(target, ast.Attribute) and target.attr == 'forward', str(path))
        self.assertEqual(list(EXTERNAL.rglob('*.patch')), [])

    def test_all_bundled_models_verify_without_git_or_network(self):
        with patch.object(INSTALL, 'git', side_effect=AssertionError('Unexpected Git/network access')):
            for name, row in INSTALL.specifications().items():
                if row['bundled']:
                    source = INSTALL.verify_source(name)
                    self.assertEqual(source, (EXTERNAL / row['directory'] / 'upstream').resolve())
                    self.assertTrue((EXTERNAL / row['directory'] / 'interface.py').is_file())

    def snapshot(self, root):
        directory = (root / 'upstream').resolve()
        directory.mkdir()
        content = b'original = True\n'
        (directory / 'model.py').write_bytes(content)
        expected = dict(sha256=hashlib.sha256(content).hexdigest(),
            git_blob=hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest())
        row = INSTALL.specifications()['TrajCL']
        manifest = root / 'SOURCE.json'
        value = dict(version=1, model='TrajCL', commit=row['commit'], repository=row['repository'],
                     files={'model.py': expected})
        manifest.write_text(json.dumps(value))
        return directory, row, manifest, value

    def test_modified_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, row, manifest, _ = self.snapshot(Path(folder))
            (directory / 'model.py').write_text('replacement = True\n')
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                INSTALL.verify_snapshot(directory, 'TrajCL', row, manifest)

    def test_missing_snapshot_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, row, manifest, _ = self.snapshot(Path(folder))
            (directory / 'model.py').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'missing author source'):
                INSTALL.verify_snapshot(directory, 'TrajCL', row, manifest)

    def test_added_snapshot_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, row, manifest, _ = self.snapshot(Path(folder))
            (directory / 'replacement.py').write_text('pass\n')
            with self.assertRaisesRegex(RuntimeError, 'untracked source'):
                INSTALL.verify_snapshot(directory, 'TrajCL', row, manifest)

    def test_snapshot_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, row, manifest, value = self.snapshot(Path(folder))
            value['files'] = {'../outside.py': value['files']['model.py']}
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'Invalid bundled source path'):
                INSTALL.verify_snapshot(directory, 'TrajCL', row, manifest)

    def test_snapshot_provenance_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, row, manifest, value = self.snapshot(Path(folder))
            value['commit'] = '0' * 40
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, 'provenance mismatch'):
                INSTALL.verify_snapshot(directory, 'TrajCL', row, manifest)

    def test_optional_sources_are_not_required_by_default(self):
        with patch.object(INSTALL, 'git', side_effect=AssertionError('Unexpected clone')):
            self.assertEqual(INSTALL.main(['--verify-only']), 0)

    def test_per_model_interfaces_delegate_to_the_matching_encoder(self):
        shared = ModuleType('external_models.reproduction.sdd_porto.sdd_models')
        calls = []
        shared.construct = lambda name, metric: calls.append((name, metric)) or (name, metric)
        with patch.dict(sys.modules, {shared.__name__: shared}):
            for name, row in INSTALL.specifications().items():
                if row['directory'].startswith('learned/'):
                    interface = load('leap_interface_' + name, row['directory'] + '/interface.py')
                    self.assertEqual(interface.construct('dtw'), (name, 'dtw'))
        self.assertEqual(len(calls), 6)

    def test_hnsw_control_uses_the_native_faiss_boundary(self):
        from unittest.mock import Mock
        native = ModuleType('faiss')
        native.METRIC_L2 = 1
        native.omp_set_num_threads = Mock()
        index = Mock(d=128, ntotal=1000)
        native.IndexHNSWFlat = Mock(return_value=index)
        arrays = ModuleType('numpy')
        arrays.float32 = object()
        values = Mock(ndim=2, shape=(1000, 128))
        values.__len__ = Mock(return_value=1000)
        arrays.ascontiguousarray = Mock(return_value=values)
        arrays.isfinite = Mock(return_value=Mock(all=lambda: True))
        interface = load('leap_hnsw_boundary', 'indexes/HNSW/interface.py')
        with patch.dict(sys.modules, faiss=native, numpy=arrays):
            self.assertIs(interface.build(values), index)
            native.IndexHNSWFlat.assert_called_once_with(128, 32, 1)
            self.assertEqual(index.hnsw.efConstruction, 200)
            native.omp_set_num_threads.assert_called_with(1)
            interface.search(index, values, top_k=20, ef=80)
            native.omp_set_num_threads.assert_called_with(4)
            self.assertEqual(index.hnsw.efSearch, 80)
            index.search.assert_called_once_with(values, 20)


if __name__ == '__main__':
    unittest.main()
