"""Convert read-only triangular labels once and audit raw reference semantics."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from common import save, sha, symmetric


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--asset',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--prefix',default='porto_20200')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.geometry) as f:
        points,lengths=f['points'],f['lengths']
    asset=json.loads(args.asset.read_text())
    p=(points-np.asarray(asset['geometry_mean']))/np.asarray(asset['geometry_std'])
    offsets=np.r_[0,np.cumsum(lengths)]
    paths=[np.ascontiguousarray(p[a:b],dtype=np.float64) for a,b in zip(offsets[:-1],offsets[1:])]
    from traj_dist.distance import hausdorff,discret_frechet,dtw
    rng=np.random.default_rng(2000)
    report={}
    for metric,fn in [('hausdorff',hausdorff),('discret_frechet',discret_frechet),('dtw',dtw)]:
        source=args.data_root/f'{args.prefix}_traj_simi_dict_{metric}.pkl'
        manifest=args.output/f'{metric}.json'
        if manifest.exists():
            report[metric]={'status':'existing','manifest':str(manifest)}
            continue
        if not source.exists():
            report[metric]={'status':'blocked_missing_training_teacher','source':str(source)}
            continue
        with source.open('rb') as f:
            raw=pickle.load(f)
        checks=[]
        files=[]
        for split,(matrix,start) in enumerate(zip(raw[:3],[0,7000,8000])):
            matrix=symmetric(matrix)
            for _ in range(16):
                a,b=rng.choice(len(matrix),2,replace=False)
                actual=float(fn(paths[start+a],paths[start+b]))
                expected=float(matrix[a,b])
                if not np.isclose(actual,expected,rtol=2e-6,atol=2e-7):
                    raise ValueError(f'{metric}: raw geometry/teacher mismatch {start+a},{start+b}: {actual} != {expected}')
                checks.append({'split':split,'a':int(a),'b':int(b),'error':abs(actual-expected)})
            name=f'{metric}.{split}.npy'
            np.save(args.output/name,matrix)
            files.append(name)
        save(manifest,dict(source=str(source),source_sha256=sha(source),maximum=float(raw[3]),
                           matrices=files,geometry_sha256=sha(args.geometry),
                           normalization_asset_sha256=sha(args.asset),raw_pair_checks=checks,
                           metric=metric,split=[7000,1000,2000]))
        report[metric]={'status':'verified','manifest':str(manifest),'raw_pair_checks':len(checks)}
        print(json.dumps(report[metric]),flush=True)
        del raw,matrix
    save(args.output/'status.json',report)


if __name__=='__main__':
    main()
