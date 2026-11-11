"""Run an author Python entry point unchanged in a separate process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from install_sources import DEFAULT_SOURCE_ROOT, verify_source

HERE = Path(__file__).resolve().parent


def main(argv=None):
    registry = json.loads((HERE / 'registry.json').read_text(encoding='utf-8'))
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        divider = argv.index('--')
        wrapper, native = argv[:divider], argv[divider + 1:]
    else:
        wrapper, native = argv, []
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('model', choices=list(registry))
    parser.add_argument('--entrypoint', type=Path)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args(wrapper)
    entry = registry[args.model]
    if entry['status'] == 'author_implementation_missing':
        parser.error(entry['note'])
    if entry['runtime'] != 'python':
        parser.error('Use the author native JVM build/run entry points for ' + args.model)
    if args.entrypoint is None:
        parser.error('--entrypoint is required; pass author arguments after --')
    root = verify_source(args.model, args.source_root)
    script = (root / args.entrypoint).resolve()
    if not script.is_relative_to(root) or script.suffix != '.py' or not script.is_file():
        parser.error('Entry point must be an existing Python file inside the author checkout')
    return subprocess.run([sys.executable, str(script), *native], cwd=root).returncode


if __name__ == '__main__':
    raise SystemExit(main())
