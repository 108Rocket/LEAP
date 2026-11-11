"""Pack already selected original trajectories without changing their geometry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_LEAP'))
from leap.checkpoints import resolve


def load_npz(path):
    """Read either packed format and retain the source row mapping, if supplied."""
    with np.load(path, allow_pickle=False) as data:
        if 'points' not in data or not {'lengths', 'offsets'}.intersection(data.files):
            raise ValueError('NPZ requires points and either lengths or offsets')
        points = data['points'].copy()
        offsets = data['offsets'].copy() if 'offsets' in data else None
        if 'lengths' in data:
            lengths = data['lengths']
            if (points.ndim != 2 or lengths.ndim != 1 or lengths.dtype.kind not in 'iu'
                    or not len(lengths) or (lengths < 2).any()
                    or sum(map(int, lengths)) != len(points)):
                raise ValueError('lengths must be integers >=2 summing to the point count')
            # Validate the total before int64 accumulation to avoid integer overflow.
            derived = np.r_[np.int64(0), np.cumsum(lengths, dtype=np.int64)]
            if offsets is not None and not np.array_equal(offsets, derived):
                raise ValueError('lengths and offsets describe different trajectories')
            if offsets is None:
                offsets = derived
        source_ids = data['source_ids'].copy() if 'source_ids' in data else None
    return points, offsets, source_ids


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    source_ids = None
    if args.input.suffix == '.npz':
        try:
            points, offsets, source_ids = load_npz(args.input)
        except ValueError as error:
            parser.error(str(error))
    elif args.input.suffix == '.json':
        rows = json.loads(args.input.read_text(encoding='utf-8'))
        if not isinstance(rows,list) or not rows:
            parser.error('Expected a nonempty JSON list of trajectories')
        arrays = [np.asarray(row,dtype=np.float64) for row in rows]
        if any(row.ndim != 2 or row.shape[1] != 2 or len(row)<2 for row in arrays):
            parser.error('Each trajectory must have shape [N,2] with N>=2')
        points = np.concatenate(arrays)
        offsets = np.r_[0,np.cumsum([len(row) for row in arrays],dtype=np.int64)]
    else:
        parser.error('Input must be NPZ or JSON')
    if (points.ndim!=2 or points.shape[1]!=2 or points.dtype.kind!='f'
            or not np.isfinite(points).all() or offsets.ndim!=1 or offsets.dtype.kind not in 'iu'
            or len(offsets)<2 or offsets[0]!=0 or offsets[-1]!=len(points)
            or (offsets < 0).any() or (offsets > len(points)).any()
            or (np.diff(offsets.astype(np.int64))<2).any()):
        parser.error('Invalid finite packed geometry')
    if source_ids is not None and (source_ids.ndim != 1 or len(source_ids) != len(offsets)-1
                                   or source_ids.dtype.kind not in 'iu'):
        parser.error('source_ids must contain one integer source ID per trajectory')
    entry = resolve(args.checkpoint)
    asset = json.loads(entry['asset_path'].read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez(stream,points=points,offsets=offsets.astype(np.int64),
                 geometry_mean=np.asarray(asset['geometry_mean'],np.float64),
                 geometry_std=np.asarray(asset['geometry_std'],np.float64),
                 **({'source_ids': source_ids} if source_ids is not None else {}))
    print(json.dumps(dict(output=str(args.output),trajectories=len(offsets)-1,points=len(points),
                          checkpoint=args.checkpoint,geometry_transformation='none')))


if __name__ == '__main__':
    main()
