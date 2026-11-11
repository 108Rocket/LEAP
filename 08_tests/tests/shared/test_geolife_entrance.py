"""Shared setup: Geolife quick start, verified public download and ZIP inputs."""
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import pytest

from leap.launch import files, wizard


def test_default_uses_geolife_without_training_prompt():
    args = wizard.parser().parse_args([])
    with patch("builtins.input") as prompt:
        wizard.choose_mode(args, [])
        prompt.assert_not_called()
    assert args.dataset == "geolife" and args.mode == "checkpoint"
    assert wizard.DATASETS[args.dataset]["queries"] == 1000
    assert wizard.DATASETS[args.dataset]["database"] == 9386
    with pytest.raises(SystemExit):
        wizard.main(["--mode", "train"])


def test_bundled_checkpoint_matches_geolife():
    args = wizard.parser().parse_args([])
    with patch.object(wizard, "download") as network:
        asset = wizard.checkpoint(args, Path("unused"))
        network.assert_not_called()
    assert json.loads(asset.read_text())["city"] == "Geolife"
    assert files.sha256(asset) == "3505df3968321c935dd69239d0af8de745ac94c698ab2335206fa8fd7ef40325"


@pytest.mark.parametrize("corrupt", [False, True])
def test_public_download_without_account_cookies(tmp_path, corrupt):
    payload = b"test archive"
    target = tmp_path / "data.zip"

    def fetch(url, output, **kwargs):
        assert url == files.GEOLIFE_URL
        assert kwargs["use_cookies"] is False and kwargs["verify"] is True
        assert kwargs["resume"] is True
        assert kwargs["timeout"] == (15, 30) and kwargs["retries"] == 1
        Path(output).write_bytes(b"bad" if corrupt else payload)
        return output

    downloader = Mock(side_effect=fetch)
    with patch.dict(sys.modules, {"gdown": SimpleNamespace(download=downloader)}), \
            patch.object(files, "GEOLIFE_BYTES", len(payload)), \
            patch.object(files, "GEOLIFE_SHA256", hashlib.sha256(payload).hexdigest()), \
            patch.object(files, "require_space"):
        if corrupt:
            with pytest.raises(ValueError, match="verification"):
                files.download_geolife(target)
            assert not target.exists()
        else:
            assert files.download_geolife(target).read_bytes() == payload
            assert files.download_geolife(target) == target
            downloader.assert_called_once()
            target.write_bytes(b"changed")
            with pytest.raises(ValueError, match="wrong SHA256"):
                files.download_geolife(target)
            downloader.assert_called_once()


@pytest.fixture
def tiny_archive(tmp_path):
    archive = tmp_path / "data.zip"
    payloads = {name: name.encode() for name in files.GEOLIFE_MEMBERS}
    entries = {name: (len(value), hashlib.sha256(value).hexdigest()) for name, value in payloads.items()}
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in payloads.items():
            bundle.writestr("SDD_Geolife_Hausdorff/Geolife/" + name, value)
        bundle.writestr("../../outside", b"not extracted")
    with patch.object(files, "GEOLIFE_MEMBERS", entries), patch.object(files, "require_space"):
        yield archive, payloads


def extract_fixture(archive, target):
    with patch.object(files, "GEOLIFE_BYTES", archive.stat().st_size), \
            patch.object(files, "GEOLIFE_SHA256", files.sha256(archive)):
        return files.extract_geolife(archive, target)


def test_zip_extract_and_verified_reuse(tmp_path, tiny_archive):
    archive, payloads = tiny_archive
    target = tmp_path / "data"
    assert extract_fixture(archive, target) == target
    assert extract_fixture(archive, target) == target
    for name, value in payloads.items():
        assert (target / name).read_bytes() == value
    assert not (tmp_path / "outside").exists()
    (target / "base_list").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        extract_fixture(archive, target)


def test_zip_rejects_duplicate_entries(tmp_path, tiny_archive):
    archive, _ = tiny_archive
    with zipfile.ZipFile(archive, "a") as bundle, pytest.warns(UserWarning):
        bundle.writestr("SDD_Geolife_Hausdorff/Geolife/base_list", b"duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        extract_fixture(archive, tmp_path / "data")


def test_zip_rejects_member_symlink(tmp_path, tiny_archive):
    archive, payloads = tiny_archive
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in payloads.items():
            info = zipfile.ZipInfo("SDD_Geolife_Hausdorff/Geolife/" + name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, value)
    with pytest.raises(ValueError, match="Unexpected"):
        extract_fixture(archive, tmp_path / "data")


def test_geolife_source_requires_download_consent(tmp_path):
    args = wizard.parser().parse_args([])
    with patch.object(wizard, "ROOT", tmp_path), patch("builtins.input", return_value="n"), \
            patch.object(wizard, "download_geolife") as network:
        with pytest.raises(wizard.Declined):
            wizard.source_data(args)
        network.assert_not_called()


def test_local_zip_never_calls_network(tmp_path):
    archive = tmp_path / "data.zip"
    archive.write_bytes(b"fixture")
    args = wizard.parser().parse_args(["--yes", "--archive", str(archive)])
    with patch.object(wizard, "ROOT", tmp_path), patch.object(wizard, "extract_geolife") as extract, \
            patch.object(wizard, "download_geolife") as network:
        wizard.source_data(args)
        network.assert_not_called()
        extract.assert_called_once_with(archive, tmp_path / "04_data/sdd/geolife/source")


def test_dataset_caches_are_separate():
    geolife = wizard.parser().parse_args([])
    porto = wizard.parser().parse_args(["--dataset", "porto"])
    assert wizard.cache_path(geolife) != wizard.cache_path(porto)
    assert "geolife" in wizard.cache_path(geolife).parts
