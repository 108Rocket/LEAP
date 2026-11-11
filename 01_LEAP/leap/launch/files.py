"""Verified, resumable downloads and narrowly scoped dataset extraction."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tarfile
import time
import zipfile
from urllib.parse import urlsplit
import urllib.request

SDD_REVISION = "e3716164e22a94384b34e7eef96066c37206c968"
SDD_SHA256 = "7118377e98e3177ba04292f5499fc973c3afe698a6e781f41cdf62ac4b42228f"
SDD_BYTES = 28156531889
SDD_URL = ("https://huggingface.co/datasets/Airchang/SDD/resolve/"
           + SDD_REVISION + "/SequenceDistanceDataset.tar.gz")
RELEASE_REVISION = "main"
MEMBERS = {"base_list": 2044457045, "query_list": 629414,
           "Haus/haus_test_distance_matrix_result": 6390316169,
           "train_list": 3740835, "Haus/haus_train_distance_matrix_result": 72000163}
GEOLIFE_URL = "https://drive.google.com/file/d/1a-CzXT0ud6JcAKAfplpej_nyHtuDEM8T/view?usp=sharing"
GEOLIFE_ARCHIVE = "SDD_Geolife_Hausdorff.zip"
GEOLIFE_SHA256 = "c44a0d2a4fb01653300a6f2a96cd1ea0460f71526d902e8af97f23ecac989ef0"
GEOLIFE_BYTES = 210852723
GEOLIFE_MEMBERS = {
    "base_list": (107006147, "a5493daabce4fb546bc029f3af983c225b75234153955dcd96e43ae988cb2f1b"),
    "query_list": (11307299, "f48f5319f99f74a4b4e8e914229204ca572a3c01574add96b9649ce1bce426fd"),
    "train_list": (34164057, "2cb9444c52eb1bc4687e7ee6fdaff6ccafdaed0c105a46d70e6e88ee250f92ec"),
    "Haus/haus_train_distance_matrix_result": (72000163, "54a0abe2dfbba8738dd56108b52622898541dd27745ab54762a88f4205d2f970"),
    "Haus/haus_test_distance_matrix_result": (75088163, "5adad7811af0d0344e8c2ad957fca94dc9f0ee3db744b4c35008465e703c8416"),
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".tmp")
    part.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    part.replace(path)


def require_space(folder, size):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    if free < size:
        raise OSError(f"Need {size / 1e9:.1f} GB free in {folder}; available {free / 1e9:.1f} GB")


def download(url, path, expected, size=None):
    """Keep interrupted bytes, but never publish a file before checking its hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) == expected:
            return path
        raise ValueError(f"Existing file has the wrong SHA256: {path}. Move it aside before retrying.")
    part = path.with_name(path.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if size and offset > size:
        raise ValueError(f"Oversized partial download: {part}")
    if size:
        require_space(path.parent, max(0, size - offset) + 1024**3)
    if size is None or offset != size:
        print("  Connecting to " + urlsplit(url).netloc + "...", flush=True)
        headers = {"User-Agent": "LEAP-reproduction/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            if response.status == 206:
                prefix = f"bytes {offset}-"
                if not response.headers.get("Content-Range", "").startswith(prefix):
                    raise ValueError("Server returned an unexpected resume range")
            elif response.status == 200:
                offset = 0
            else:
                raise OSError(f"Unexpected HTTP status {response.status}")
            last = time.monotonic()
            with part.open("ab" if offset else "wb") as stream:
                while block := response.read(8 * 1024 * 1024):
                    offset += len(block)
                    if size and offset > size:
                        raise ValueError("Download exceeded its published size")
                    stream.write(block)
                    if time.monotonic() - last >= 10:
                        print(f"  Downloaded {offset / 1e9:.2f} GB", flush=True)
                        last = time.monotonic()
    if size and part.stat().st_size != size:
        raise ValueError(f"Incomplete download; rerun to resume {part}")
    print("  Verifying download SHA256...", flush=True)
    if sha256(part) != expected:
        raise ValueError(f"Download SHA256 mismatch: {part}. Move it aside before retrying.")
    part.replace(path)
    return path


def download_geolife(path):
    """Fetch only the public data file, without reading or saving account cookies."""
    import gdown
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != GEOLIFE_SHA256:
            raise ValueError("Existing Geolife ZIP has the wrong SHA256; it will not be overwritten")
        return path
    require_space(path.parent, GEOLIFE_BYTES + 1024**3)
    pending = path.with_name(path.name + ".pending")
    if not pending.exists():
        print("  Google Drive needs network access; alternatively use --archive PATH.zip.", flush=True)
        result = gdown.download(GEOLIFE_URL, str(pending), resume=True,
                                use_cookies=False, verify=True, quiet=False,
                                timeout=(15, 30), retries=1)
        if result is None:
            raise RuntimeError("Public download did not finish. Check link access or pass --archive PATH.zip")
    if pending.stat().st_size != GEOLIFE_BYTES or sha256(pending) != GEOLIFE_SHA256:
        raise ValueError("Downloaded Geolife ZIP failed size/SHA256 verification; retain it for inspection")
    pending.replace(path)
    return path


def verify_geolife(source):
    source = Path(source)
    for name, (size, digest) in GEOLIFE_MEMBERS.items():
        path = source / name
        if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
            raise ValueError("Missing or changed original Geolife input: " + str(path))
    return source


def extract_geolife(archive, target):
    """Extract the five allowlisted inputs; reject links and incomplete archives."""
    archive, target = Path(archive), Path(target)
    if archive.stat().st_size != GEOLIFE_BYTES or sha256(archive) != GEOLIFE_SHA256:
        raise ValueError("Geolife archive size/SHA256 mismatch")
    require_space(target, sum(value[0] for value in GEOLIFE_MEMBERS.values()) + 256 * 1024**2)
    prefix = "SDD_Geolife_Hausdorff/Geolife/"
    with zipfile.ZipFile(archive) as bundle:
        names = [item.filename for item in bundle.infolist()]
        for relative, (size, digest) in GEOLIFE_MEMBERS.items():
            name = prefix + relative
            if names.count(name) != 1:
                raise ValueError("Missing or duplicate Geolife archive entry: " + name)
            member = bundle.getinfo(name)
            if member.file_size != size or member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unexpected Geolife archive entry: " + name)
            dest = target / relative
            part = dest.with_name(dest.name + ".part")
            if any(p.is_symlink() or not p.resolve().is_relative_to(target.resolve()) for p in (dest, part)):
                raise ValueError("Extraction destination escapes its data directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if sha256(dest) != digest:
                    raise ValueError("Existing Geolife input differs; it will not be overwritten: " + str(dest))
                continue
            print("  Extracting Geolife/" + relative, flush=True)
            with bundle.open(member) as source, part.open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if sha256(part) != digest:
                raise ValueError("Geolife member SHA256 mismatch: " + relative)
            part.replace(dest)
    verify_geolife(target)
    save(target / "SOURCE.json", {"url": GEOLIFE_URL, "archive_sha256": GEOLIFE_SHA256,
                                 "files": {name: value[1] for name, value in GEOLIFE_MEMBERS.items()}})
    return target


def extract_porto(archive, target, *, training=False):
    target = Path(target)
    required = set(MEMBERS) if training else set(MEMBERS) - {"train_list", "Haus/haus_train_distance_matrix_result"}
    require_space(target, sum(MEMBERS[name] for name in required if not (target / name).is_file()) + 2 * 1024**3)
    remaining = set(required)
    seen = set()
    prefix = "SequenceDistanceDataset/Trajectory/Porto/"
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            name = member.name.removeprefix("./")
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix):]
            if relative not in required:
                continue
            if relative in seen or not member.isfile() or member.size != MEMBERS[relative]:
                raise ValueError("Unexpected SDD archive entry: " + member.name)
            seen.add(relative)
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink() or not dest.resolve().is_relative_to(target.resolve()):
                raise ValueError("Extraction destination escapes its data directory")
            if dest.exists():
                with bundle.extractfile(member) as source:
                    value = hashlib.sha256()
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        value.update(block)
                if sha256(dest) != value.hexdigest():
                    raise ValueError(f"Existing extracted file differs from the verified archive: {dest}")
                remaining.remove(relative)
                if not remaining:
                    break
                continue
            print("  Extracting Porto/" + relative, flush=True)
            with bundle.extractfile(member) as source, dest.with_name(dest.name + ".part").open("wb") as output:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            dest.with_name(dest.name + ".part").replace(dest)
            remaining.remove(relative)
            if not remaining:
                break
    if remaining:
        raise ValueError("Archive is missing Porto inputs: " + ", ".join(sorted(remaining)))
    save(target / "SOURCE.json", {"url": SDD_URL, "archive_sha256": SDD_SHA256,
                                  "files": {name: sha256(target / name) for name in sorted(required)}})
    return target
