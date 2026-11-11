"""RQ1/RQ2: safe guided setup, original-data references and timing configuration."""
import ast
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile
from unittest.mock import patch

import numpy as np
import pytest

from leap.launch import data, files, wizard
from leap.launch.train import nearfar_source, Sizes


def test_defaults():
    args = wizard.parser().parse_args([])
    assert args.mode == "checkpoint" and args.scope == "both" and not args.yes
    assert args.dataset == "geolife"


@pytest.mark.parametrize("answer,expected", [("", "checkpoint"), ("Y", "checkpoint"), ("n", "train")])
def test_interactive_model_choice(answer, expected):
    args = wizard.parser().parse_args(["--dataset", "porto"])
    with patch.object(wizard.sys.stdin, "isatty", return_value=True), patch("builtins.input", return_value=answer):
        wizard.choose_mode(args, [])
    assert args.mode == expected


def test_environment_restart_keeps_resolved_paths(tmp_path):
    source = tmp_path / "data with spaces"
    args = wizard.parser().parse_args(["--dataset", "porto", "--source-dir", str(source), "--mode", "train", "--yes"])
    command = wizard.resume_command(args, tmp_path / "bin/python")
    assert command[command.index("--source-dir") + 1] == source
    assert command[command.index("--mode") + 1] == "train" and command[-1] == "--yes"
    assert command[command.index("--dataset") + 1] == "porto"


@pytest.mark.parametrize("answer", ["", "n", "no", "YES PLEASE"])
def test_consent_defaults_no(answer):
    with patch("builtins.input", return_value=answer), pytest.raises(wizard.Declined):
        wizard.confirm("Download?")


def test_consent_eof():
    with patch("builtins.input", side_effect=EOFError), pytest.raises(wizard.Declined):
        wizard.confirm("Download?")


def test_prompt_visible_before_waiting_for_input(capsys):
    def answer():
        assert capsys.readouterr().out == "Download? [y/N]\n"
        return "y"
    with patch("builtins.input", side_effect=answer):
        wizard.confirm("Download?")


@pytest.mark.parametrize("answer", ["y", "Y", "yes"])
def test_consent_yes(answer):
    with patch("builtins.input", return_value=answer):
        wizard.confirm("Download?")


def test_numeric_pickle(tmp_path):
    path = tmp_path / "numeric"
    original = [np.array([[1., 2.], [3., 4.]])]
    path.write_bytes(pickle.dumps(original, protocol=4))
    np.testing.assert_array_equal(data.numeric_load(path)[0], original[0])
    path.write_bytes(pickle.dumps(Path("forbidden")))
    with pytest.raises(pickle.UnpicklingError):
        data.numeric_load(path)


def test_stable_top_and_hd():
    np.testing.assert_array_equal(data.top_ids(np.array([2., 1., 1., 3.]), 2), [1, 2])
    assert data.pointset_hd(np.array([[0., 0.], [1., 0.]]), np.array([[0., 0.], [3., 0.]])) == 2.
    with pytest.raises(ValueError):
        data.top_ids(np.array([np.nan, 1.]), 1)


def test_prepare_and_hash_checked_reuse(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "cache"
    (source / "Haus").mkdir(parents=True)
    base = [np.array([[i, 0.], [i + 1., 0.]]) for i in range(24)]
    queries = base[:2]
    matrix = np.array([[data.pointset_hd(q, b) for b in base] for q in queries])
    for name, value in (("base_list", base), ("query_list", queries), ("Haus/haus_test_distance_matrix_result", matrix)):
        (source / name).write_bytes(pickle.dumps(value, protocol=4))
    asset = tmp_path / "asset.json"
    asset.write_text(json.dumps({"geometry_mean": [0., 0.], "geometry_std": [1., 1.]}))
    first = data.prepare(source, dest, asset, expected_database=24, expected_queries=2)
    second = data.prepare(source, dest, asset, expected_database=24, expected_queries=2)
    assert first == second
    with np.load(dest / "database.npz", allow_pickle=False) as packed:
        np.testing.assert_array_equal(packed["points"], np.concatenate(base))
    (dest / "top20.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted"):
        data.prepare(source, dest, asset, expected_database=24, expected_queries=2)


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status, self.headers = status, headers or {}


@pytest.mark.parametrize("resume", [False, True])
def test_atomic_download(tmp_path, resume):
    payload = b"verified payload"
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "download"
    if resume:
        dest.with_name("download.part").write_bytes(payload[:4])
    response = Response(payload[4:] if resume else payload, 206 if resume else 200,
                        {"Content-Range": f"bytes 4-{len(payload)-1}/{len(payload)}"})
    with patch.object(files.urllib.request, "urlopen", return_value=response), patch.object(files, "require_space"):
        files.download("https://example.invalid/file", dest, digest, len(payload))
    assert dest.read_bytes() == payload


def test_download_bad_hash_not_published(tmp_path):
    dest = tmp_path / "download"
    with patch.object(files.urllib.request, "urlopen", return_value=Response(b"wrong")):
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            files.download("https://example.invalid/file", dest, "0" * 64)
    assert not dest.exists() and dest.with_name("download.part").exists()


def test_existing_bad_download_not_overwritten(tmp_path):
    dest = tmp_path / "download"
    dest.write_bytes(b"existing")
    with patch.object(files.urllib.request, "urlopen") as network:
        with pytest.raises(ValueError, match="wrong SHA256"):
            files.download("https://example.invalid/file", dest, "0" * 64)
        network.assert_not_called()
    assert dest.read_bytes() == b"existing"


def test_extract_rejects_links(tmp_path):
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        member = tarfile.TarInfo("SequenceDistanceDataset/Trajectory/Porto/base_list")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escape"
        out.addfile(member)
    with patch.object(files, "require_space"), pytest.raises(ValueError, match="Unexpected"):
        files.extract_porto(archive, tmp_path / "output")


def test_extract_resumes_only_identical_files(tmp_path):
    archive = tmp_path / "data.tar.gz"
    members = {"base_list": b"base", "query_list": b"query",
               "Haus/haus_test_distance_matrix_result": b"labels"}
    with tarfile.open(archive, "w:gz") as out:
        for name, payload in members.items():
            member = tarfile.TarInfo("SequenceDistanceDataset/Trajectory/Porto/" + name)
            member.size = len(payload)
            out.addfile(member, io.BytesIO(payload))
        malicious = tarfile.TarInfo("../../outside")
        malicious.size = 3
        out.addfile(malicious, io.BytesIO(b"bad"))
    dest = tmp_path / "output"
    dest.mkdir()
    (dest / "base_list").write_bytes(b"base")
    with patch.object(files, "MEMBERS", {k: len(v) for k, v in members.items()}), patch.object(files, "require_space"):
        files.extract_porto(archive, dest)
        assert (dest / "SOURCE.json").is_file()
        assert not (tmp_path / "outside").exists()
        (dest / "base_list").write_bytes(b"changed")
        with pytest.raises(ValueError, match="differs"):
            files.extract_porto(archive, dest)


def test_checkpoint_recovered_from_git_without_network(tmp_path):
    root, run = tmp_path / "code", tmp_path / "run"
    (root / "03_checkpoints").mkdir(parents=True)
    run.mkdir()
    blob = b'{"anchors": []}'
    digest = hashlib.sha256(blob).hexdigest()
    (root / "03_checkpoints/manifest.json").write_text(json.dumps({"checkpoints": [
        {"id": "sdd.porto.hausdorff", "asset": "03_checkpoints/missing.json", "asset_sha256": digest}]}))
    args = wizard.parser().parse_args(["--dataset", "porto", "--yes"])
    import subprocess
    completed = subprocess.CompletedProcess([], 0, stdout=blob)
    with patch.object(wizard, "ROOT", root), patch.object(wizard.subprocess, "run", return_value=completed), \
            patch.object(wizard, "download") as network:
        assert wizard.checkpoint(args, run).read_bytes() == blob
        network.assert_not_called()


def test_checkpoint_download_uses_current_repository(tmp_path):
    root, run = tmp_path / "code", tmp_path / "run"
    (root / "03_checkpoints").mkdir(parents=True)
    run.mkdir()
    blob = b'{"anchors": []}'
    digest = hashlib.sha256(blob).hexdigest()
    (root / "03_checkpoints/manifest.json").write_text(json.dumps({"checkpoints": [
        {"id": "sdd.porto.hausdorff", "asset": "03_checkpoints/missing.json", "asset_sha256": digest}]}))
    args = wizard.parser().parse_args(["--dataset", "porto", "--yes"])

    def fetch(url, target, expected):
        assert url == ("https://raw.githubusercontent.com/DongXinfzu/LEAP/"
                       + files.RELEASE_REVISION + "/03_checkpoints/missing.json")
        assert expected == digest
        target.write_bytes(blob)

    with patch.object(wizard, "ROOT", root), \
            patch.object(wizard.subprocess, "run", side_effect=FileNotFoundError), \
            patch.object(wizard, "download", side_effect=fetch) as network:
        assert wizard.checkpoint(args, run).read_bytes() == blob
        network.assert_called_once()


def test_custom_checkpoint_requires_completed_training(tmp_path):
    asset = tmp_path / "final_asset.json"
    asset.write_text(json.dumps({"seed": 2000, "city": "Porto", "representation": "pointset-nearfar-v1"}))
    digest = files.sha256(asset)
    (tmp_path / "ALL_DONE.json").write_text(json.dumps({"status": "trained", "epochs": 100,
                                                      "asset_sha256": digest}))
    args = wizard.parser().parse_args(["--dataset", "porto", "--checkpoint", str(asset)])
    assert wizard.checkpoint(args, tmp_path) == asset
    (tmp_path / "ALL_DONE.json").write_text(json.dumps({"status": "running", "epochs": 1,
                                                      "asset_sha256": digest}))
    with pytest.raises(ValueError, match="completed"):
        wizard.checkpoint(args, tmp_path)


def test_train_cannot_silently_use_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        wizard.main(["--mode", "train", "--checkpoint", str(tmp_path / "asset.json")])


def test_download_rejects_wrong_resume_range(tmp_path):
    path = tmp_path / "download"
    path.with_name("download.part").write_bytes(b"first")
    with patch.object(files.urllib.request, "urlopen", return_value=Response(b"rest", 206, {"Content-Range": "bytes 2-8/9"})):
        with pytest.raises(ValueError, match="range"):
            files.download("https://example.invalid/file", path, "0" * 64)
    assert not path.exists()


def test_training_adapter_matches_same_field_formula():
    source = wizard.ROOT / "06_reproduction/training_sources"
    for name in ("select_probes", "place"):
        adapted = nearfar_source(name, (source / (name + ".py")).read_text())
        ast.parse(adapted)
        assert ".abs()" in adapted
    with pytest.raises(ValueError):
        nearfar_source("place", "unrecognized source")


def test_training_size_adapter():
    parsed = ast.fix_missing_locations(Sizes(2400, 600).visit(ast.parse("x = [7000,1000,2000]\ny=5500")))
    env = {}
    exec(compile(parsed, "fixture", "exec"), env)
    assert env["x"] == [2400, 600, 0] and env["y"] == 1900


def test_check_only_never_runs_gpu_or_download(tmp_path):
    with patch.object(wizard, "environment") as gpu, patch.object(wizard, "download") as network:
        assert wizard.main(["--check-only", "--log-root", str(tmp_path)]) == 0
        gpu.assert_not_called()
        network.assert_not_called()
    run = next(tmp_path.iterdir())
    assert (run / "console.log").is_file() and not (run / "ALL_DONE.json").exists()


def test_declined_logs_do_not_claim_success(tmp_path):
    with patch.object(wizard, "environment", side_effect=wizard.Declined("declined")):
        assert wizard.main(["--log-root", str(tmp_path)]) == 2
    run = next(tmp_path.iterdir())
    assert (run / "STOPPED.json").is_file() and not (run / "ALL_DONE.json").exists()


def test_explicit_incomplete_source_no_download(tmp_path):
    args = wizard.parser().parse_args(["--source-dir", str(tmp_path)])
    with patch.object(wizard, "download") as download:
        with pytest.raises(FileNotFoundError):
            wizard.source_data(args)
        download.assert_not_called()


def test_resume_space_check_counts_existing_partial(tmp_path):
    args = wizard.parser().parse_args(["--dataset", "porto", "--yes"])
    partial = tmp_path / "04_data/downloads/SequenceDistanceDataset.tar.gz.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"12345")
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard, "require_space") as space, \
            patch.object(wizard, "download"), patch.object(wizard, "extract_porto"):
        wizard.source_data(args)
    space.assert_called_once_with(partial.parent, 45 * 10**9 - 5)


@pytest.mark.parametrize("mode", ["train", "checkpoint"])
def test_trained_model_uses_own_cache_by_default(tmp_path, mode):
    asset = tmp_path / "final_asset.json"
    asset.write_text('{"geometry_mean": [1, 2]}')
    arguments = ["--dataset", "porto", "--mode", mode]
    if mode == "checkpoint":
        arguments += ["--checkpoint", str(asset)]
    args = wizard.parser().parse_args(arguments)
    with patch.object(wizard, "ROOT", tmp_path):
        assert wizard.cache_path(args, asset).name == files.sha256(asset)
        args.cache_dir = tmp_path / "explicit-cache"
        assert wizard.cache_path(args, asset) == args.cache_dir


def test_bootstrap_requires_consent_and_uses_only_venv(tmp_path):
    args = wizard.parser().parse_args([])
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard.platform, "system", return_value="Linux"), \
            patch.object(wizard, "dependencies", return_value=["torch missing"]), \
            patch("builtins.input", return_value="n"), patch.object(wizard, "command") as commands:
        with pytest.raises(wizard.Declined):
            wizard.environment(args)
        commands.assert_not_called()
    args.yes = True
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard.platform, "system", return_value="Linux"), \
            patch.object(wizard, "dependencies", return_value=["torch missing"]), \
            patch.object(wizard, "check_package_network"), \
            patch.object(wizard, "command") as commands:
        assert wizard.environment(args) is False
        calls = [call.args[0] for call in commands.call_args_list]
        assert calls[0][1:3] == ["-m", "venv"]
        assert all(str(call[0]).startswith(str(tmp_path / ".venv-entrance")) for call in calls[1:])


def test_managed_environment_repairs_once(tmp_path):
    target = tmp_path / ".venv-entrance/bin/python"
    target.parent.mkdir(parents=True)
    target.touch()
    args = wizard.parser().parse_args(["--yes"])
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard.platform, "system", return_value="Linux"), \
            patch.object(wizard, "dependencies", return_value=["torch missing"]), \
            patch.object(wizard.sys, "prefix", str(target.parent.parent)), \
            patch.dict(wizard.os.environ, {"LEAP_BOOTSTRAPPED": "1"}), patch.object(wizard, "command") as commands:
        with pytest.raises(RuntimeError, match="still incompatible"):
            wizard.environment(args)
        commands.assert_not_called()


def test_insufficient_memory_stops_before_installation(tmp_path):
    args = wizard.parser().parse_args(["--yes"])
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard.platform, "system", return_value="Linux"), \
            patch.object(data, "check_preparation_memory", side_effect=MemoryError("insufficient RAM")), \
            patch.object(wizard, "command") as command:
        with pytest.raises(MemoryError):
            wizard.environment(args)
        command.assert_not_called()


def test_package_network_has_parent_timeout():
    import subprocess
    with patch.object(wizard.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 20)) as run:
        with pytest.raises(RuntimeError, match="wheelhouse"):
            wizard.check_package_network()
        assert run.call_args.kwargs["timeout"] == 20


def test_offline_install_uses_no_network_and_preserves_option(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "example.whl").touch()
    args = wizard.parser().parse_args(["--yes", "--wheelhouse", str(wheels)])
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard.platform, "system", return_value="Linux"), \
            patch.object(data, "check_preparation_memory"), \
            patch.object(wizard, "dependencies", return_value=["torch missing"]), \
            patch.object(wizard, "check_package_network") as network, patch.object(wizard, "command") as commands:
        assert wizard.environment(args) is False
        network.assert_not_called()
        calls = [call.args[0] for call in commands.call_args_list]
        installs = [call for call in calls if "pip" in call]
        assert len(installs) == 4
        assert all("--no-index" in call and wheels in call for call in installs)
        assert "--no-build-isolation" in installs[-1]
        assert calls[-1][calls[-1].index("--wheelhouse") + 1] == wheels
