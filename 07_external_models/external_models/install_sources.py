"""Fetch pinned author sources without applying model changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = None
SOURCE_SUFFIXES = {'.py', '.pyx', '.pyd', '.so', '.java', '.scala'}


def specifications():
    specs = json.loads((HERE / 'sources.json').read_text(encoding='utf-8'))
    for name, row in specs.items():
        if Path(name).name != name or not re.fullmatch(r'[0-9a-f]{40}', row['commit']):
            raise ValueError('Invalid source specification: ' + name)
        directory = Path(row['directory'])
        if directory.is_absolute() or '..' in directory.parts:
            raise ValueError('Invalid model directory: ' + name)
    return specs


def git(*arguments, cwd=None):
    try:
        return subprocess.run(
            ['git', '-c', 'core.autocrlf=false', *map(str, arguments)], cwd=cwd,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Author-source Git operation failed:\n' + exc.stderr.strip()) from exc


def source_directory(name, source_root=None):
    if source_root is not None:
        return (Path(source_root) / name).resolve()
    return (HERE / specifications()[name]['directory'] / 'upstream').resolve()


def verify_snapshot(directory, name, row, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('version') != 1 or manifest.get('model') != name
            or manifest.get('commit') != row['commit']
            or manifest.get('repository') != row['repository']):
        raise RuntimeError(name + ': bundled source provenance mismatch')
    if not manifest.get('files'):
        raise ValueError(name + ': empty source manifest')
    for relative, expected in manifest['files'].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Invalid bundled source path')
        if not path.is_file():
            raise FileNotFoundError(name + ': missing author source: ' + relative)
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if (hashlib.sha256(content).hexdigest() != expected['sha256']
                or blob != expected['git_blob']):
            raise RuntimeError(name + ': bundled author source changed: ' + relative)
    extras = [p for p in directory.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.suffix in SOURCE_SUFFIXES
              and p.relative_to(directory).as_posix() not in manifest['files']]
    if extras:
        raise RuntimeError(name + ': untracked source files in bundled author code')
    return directory


def verify_source(name, source_root=None):
    row = specifications()[name]
    directory = source_directory(name, source_root)
    if not directory.is_dir():
        raise FileNotFoundError('Install author source first: ' + str(directory))
    manifest = directory.parent / 'SOURCE.json'
    if source_root is None and manifest.is_file() and not (directory / '.git').exists():
        return verify_snapshot(directory, name, row, manifest)
    top = Path(git('rev-parse', '--show-toplevel', cwd=directory)).resolve()
    if top != directory:
        raise RuntimeError(name + ': expected a separate author checkout')
    head = git('rev-parse', 'HEAD', cwd=directory)
    if head != row['commit']:
        raise RuntimeError(name + ': source commit differs from sources.json')
    dirty = git('status', '--porcelain', '--untracked-files=no', cwd=directory)
    if dirty:
        raise RuntimeError(name + ': author source has tracked changes\n' + dirty)
    extras = git('ls-files', '--others', '--exclude-standard', cwd=directory).splitlines()
    if any(Path(path).suffix in SOURCE_SUFFIXES for path in extras):
        raise RuntimeError(name + ': untracked source files in author checkout')
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', action='append')
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--include-optional', action='store_true')
    args = parser.parse_args(argv)
    specs = specifications()
    registry = json.loads((HERE / 'registry.json').read_text(encoding='utf-8'))
    selected = args.model or [name for name, row in specs.items()
                             if args.include_optional or not row.get('optional')]
    for name in selected:
        if name not in specs:
            parser.error('Unknown model: ' + name)
        row = specs[name]
        if args.list:
            print(name + ' ' + row['commit'] + ' ' + row['repository'])
            continue
        directory = source_directory(name, args.source_root)
        if not directory.exists():
            if args.verify_only:
                raise FileNotFoundError(directory)
            directory.parent.mkdir(parents=True, exist_ok=True)
            git('clone', '--no-checkout', row['repository'], directory)
            git('checkout', '--detach', row['commit'], cwd=directory)
        directory = verify_source(name, args.source_root)
        status = dict(model=name, source=str(directory), commit=row['commit'],
                      unmodified=True, status=registry[name]['status'])
        if registry[name].get('note'):
            status['note'] = registry[name]['note']
        print(json.dumps(status))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
