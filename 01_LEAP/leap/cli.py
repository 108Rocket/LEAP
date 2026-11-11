"""Command line interface for review artifacts, timing, evaluation and training."""
import argparse
import json
from pathlib import Path
import sys

from .checkpoints import ROOT, catalog, resolve, verify


def emit(value):
    print(json.dumps(value,indent=2,ensure_ascii=False,default=str))


def parser():
    p = argparse.ArgumentParser(description='LEAP: trajectory similarity search and reproduction')
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('checkpoints',help='list published seed-2000 checkpoint identifiers')
    check = sub.add_parser('check',help='verify checkpoints and report dataset placement')
    check.add_argument('--config',type=Path,default=ROOT/'05_examples/reproduce_seed2000.json')
    check.add_argument('--require-data',action='store_true')
    for command in ('timing','evaluate','reproduce'):
        item = sub.add_parser(command)
        item.add_argument('--config',type=Path,default=ROOT/'05_examples/reproduce_seed2000.json')
        item.add_argument('--job',required=True)
        item.add_argument('--output',type=Path)
        item.add_argument('--dry-run',action='store_true')
        if command == 'timing':
            item.add_argument('--scope',choices=('batch','single','both'),default='both')
            item.add_argument('--query-ids',type=Path)
            item.add_argument('--memory-fraction',type=float,default=0.45)
    train = sub.add_parser('train',help='print a training recipe or explicitly execute it')
    train.add_argument('--config',type=Path,default=ROOT/'05_examples/reproduce_seed2000.json')
    train.add_argument('--recipe',required=True,choices=('paper-probes','dtw-probes','dtw-routing'))
    train.add_argument('--output',type=Path)
    train.add_argument('--execute',action='store_true')
    return p


def check_inputs(path,require_data=False):
    from .reproduction import config
    base,cfg = config.read(path)
    entries = catalog()
    for entry in entries:
        verify(entry)
    datasets = []
    for name,spec in cfg['datasets'].items():
        files = {key:config.resolve(base,spec[key]['path'] if isinstance(spec[key],dict) else spec[key])
                 for key in ('database','queries') if key in spec}
        datasets.append(dict(dataset=name,files={key:dict(path=str(file),present=file.is_file()) for key,file in files.items()}))
    for name in cfg['jobs']:
        config.plan(base,cfg,name)
    missing = any(not f['present'] for item in datasets for f in item['files'].values())
    emit(dict(checkpoints_verified=len(entries),seed=2000,datasets=datasets,
              dataset_note='Place datasets in 04_data/ as documented in 04_data/README.md.',
              data_ready=not missing,configuration_valid=True))
    return int(require_data and missing)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'checkpoints':
            for row in catalog():
                print(row['id'])
            return 0
        if args.command == 'check':
            return check_inputs(args.config,args.require_data)
        from .reproduction import config
        from .reproduction.logging import output_path, run_logged
        path,cfg = config.read(args.config)
        if args.command == 'train':
            from .reproduction.training import plan,run
            request = plan(path,cfg,args.recipe,args.output)
            if args.execute:
                run_logged(request['output'], {'command': 'train', 'configuration': cfg, 'plan': request},
                    lambda artifacts: run(dict(request, output=str(artifacts))))
                print('LOGS', request['output'])
            else:
                emit(request)
            return 0
        request = config.plan(path,cfg,args.job)
        if args.command == 'timing':
            from .reproduction.timing import prepare_plan,run
            request = prepare_plan(request,args.scope,args.query_ids,args.memory_fraction)
        else:
            from .reproduction.replay import run
        category = 'timing' if args.command == 'timing' else 'evaluation'
        destination = (args.output.expanduser().resolve() if args.output else
                       output_path(request['output_root'], category, args.job))
        request['log_directory'] = str(destination)
        if args.dry_run:
            emit(request)
        else:
            result = run_logged(destination, {'command': args.command, 'configuration': cfg, 'plan': request},
                                lambda artifacts: run(request, artifacts))
            if result is not None:
                emit(result)
            print('LOGS', destination)
        return 0
    except (OSError,ValueError,KeyError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 2
