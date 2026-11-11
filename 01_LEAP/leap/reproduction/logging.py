"""Per-run configuration, console capture and status for the public commands."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import traceback


def output_path(root, category, name):
    if category not in ('training', 'evaluation', 'timing', 'tests'):
        raise ValueError('Unknown log category')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid run name')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    return Path(root).expanduser().resolve()/category/(name+'-'+stamp)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n', encoding='utf-8')


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def run_logged(directory, settings, execute):
    """Give the worker a new artifacts directory; never reuse an existing run."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = directory/'artifacts'
    save(directory/'configuration.json', dict(settings, artifacts=str(artifacts)))
    status = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat())
    save(directory/'status.json', status)
    with (directory/'console.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                result = execute(artifacts)
            except BaseException as exc:
                status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                traceback.print_exc()
                raise
            else:
                status['status'] = 'complete'
                return result
            finally:
                status['finished_utc'] = datetime.now(timezone.utc).isoformat()
                save(directory/'status.json', status)
