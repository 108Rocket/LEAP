"""Verify the shipped paper learning sources without importing ML code."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for entry in manifest['files']:
        source = root / entry['path']
        text = source.read_text(encoding='utf-8')
        ast.parse(text, filename=entry['path'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Source byte hash mismatch: ' + entry['path'])
    print(f"Verified {len(manifest['files'])} paper learning source files.")
    print('Source-only check; no training, inference, or benchmark was executed.')


if __name__ == '__main__':
    main()
