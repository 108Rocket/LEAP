"""Run the documented GPU-free release checks on a Linux development host."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'01_LEAP'))
from leap.reproduction.logging import output_path, run_logged
CPU_TESTS = ['rq1/test_adopted_profiles.py','rq5/test_auto_config.py','rq5/test_candidate_capacity.py',
             'shared/test_data.py','rq1/test_query_batch_defaults.py','shared/test_runtime.py',
             'shared/test_timing.py','rq3/test_dtw_routing.py','shared/test_release.py',
             'rq1/test_external_interfaces.py','shared/test_logs.py']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cpu',action='store_true',help='run the GPU-free configuration and input suite')
    parser.add_argument('--output',type=Path,help='new log directory; defaults to 14_logs/tests/TIMESTAMP')
    args = parser.parse_args()
    source_roots = ('01_LEAP', '07_external_models', '08_tests')
    environment = dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/path) for path in source_roots),PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    if args.cpu:
        environment['CUDA_VISIBLE_DEVICES'] = ''
        targets = [str(ROOT/'08_tests/tests'/name) for name in CPU_TESTS]
    else:
        targets = [str(ROOT/'08_tests/tests')]
    destination = args.output or output_path(ROOT/'14_logs', 'tests', 'cpu' if args.cpu else 'full')

    def execute(artifacts):
        artifacts.mkdir()
        command = [sys.executable, '-m', 'pytest', '-q', *targets, '-p', 'no:cacheprovider',
                   '--junitxml='+str(artifacts/'junit.xml')]
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace') as process:
            for line in process.stdout:
                print(line, end='')
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code

    try:
        return run_logged(destination, dict(cpu=args.cpu, targets=targets, source_roots=source_roots), execute)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    finally:
        print('LOGS', Path(destination).resolve())


if __name__ == '__main__':
    raise SystemExit(main())
