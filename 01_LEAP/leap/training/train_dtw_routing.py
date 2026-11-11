"""Train candidate inclusion on full training DTW neighbors, never test labels."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from leap import LEAP
from leap import RetrievalConfig
from leap.dtw import dtw_routing as routing
from leap.dtw.dtw_probes import DEFAULT_ASSET, load_asset
from leap.training.train_dtw_probes import load_paths, save


def with_summary(index, summary):
    return SimpleNamespace(coordinates=index.coordinates, points=index.points,
                           offsets=index.offsets, ordered_summary=summary)


@torch.no_grad()
def encode(net, data):
    return torch.cat([net(*(x[lo:lo+512] for x in data)) for lo in range(0, len(data[1]), 512)])


@torch.no_grad()
def truth(q, db, same=False):
    from leap.dtw.dtw import dtw_pairs
    from leap.common.interval_index import lex_keys
    ids, values = [], []
    for lo in range(0, len(q.coordinates), 32):
        hi = min(lo+32, len(q.coordinates))
        rows = torch.arange(lo, hi, device=q.points.device)[:, None].expand(hi-lo, len(db.coordinates))
        cols = torch.arange(len(db.coordinates), device=q.points.device)[None].expand_as(rows)
        distance = dtw_pairs(q, db, rows.flatten(), cols.flatten()).reshape_as(rows)
        if same:
            distance[torch.arange(hi-lo, device=distance.device), torch.arange(lo,hi,device=distance.device)] = torch.inf
        selected = lex_keys(distance, cols).topk(20, largest=False).values & 0xffffffff
        ids.append(selected)
        values.append(distance.gather(1,selected))
    return torch.cat(ids), torch.cat(values)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--asset', type=Path, default=DEFAULT_ASSET)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=2000)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args(argv)
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing training is not overwritten')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    asset = load_asset(args.asset)
    trpaths, vapaths = load_paths(args.raw_inputs, 7000, 1000, asset)
    model = LEAP.from_asset(args.asset, device=args.device, config=RetrievalConfig('dtw'))
    tr, va = model.encode(trpaths), model.encode(vapaths)
    train, validation = routing.prepare(tr), routing.prepare(va)
    positive, distance = truth(tr,tr,True)
    valtruth, _ = truth(va,tr)
    radius = distance[:, -1]
    from leap.dtw.ordered_dtw import candidates
    net = routing.OrderedMass().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    history, best = [], -1.
    own = torch.arange(7000, device=args.device)
    protocol = dict(train=7000, validation_queries=1000, validation_database=7000,
                    epochs=args.epochs, seed=args.seed, batch=64, lr=.003, candidate_budget=128,
                    selection='earliest maximum validation Top20-in128 coverage', test_selection_enabled=False)
    save(args.output/'protocol.json',protocol)
    for epoch in range(args.epochs+1):
        losses = []
        if epoch:
            with torch.no_grad():
                index = with_summary(tr,encode(net,train))
                pool = candidates(index,index,256,own,max_work=True)
                ispositive = (pool[:,:,None] == positive[:,None]).any(-1)
                slots = torch.arange(256,device=args.device)[None].expand_as(pool)
                boundary = slots.masked_fill(ispositive,1000).topk(109,largest=False).values[:,-1]
                negative = pool.gather(1,boundary[:,None]).squeeze(1)
            generator = torch.Generator(device=args.device).manual_seed(args.seed+epoch)
            for rr in torch.randperm(7000,device=args.device,generator=generator).split(64):
                ds = torch.cat([positive[rr],negative[rr,None]],1)
                ids,inv = torch.unique(torch.cat([rr,ds.flatten()]),return_inverse=True)
                summary,logmass,_ = net(*(x[ids] for x in train),details=True)
                qr,dr = inv[:len(rr)],inv[len(rr):].reshape_as(ds)
                scores = routing.pair_scores(summary[qr,None],summary[dr],train[1][rr,None],train[1][ds])
                loss = routing.coverage_loss(scores[:,:20],scores[:,-1],radius[rr])
                loss = loss+1e-5*logmass.square().sum()/train[1][ids].sum()
                optimizer.zero_grad()
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                if not bool(torch.isfinite(loss)&torch.isfinite(gradient)):
                    raise FloatingPointError('Nonfinite routing loss/gradient')
                optimizer.step()
                losses.append(float(loss))
        with torch.no_grad():
            q,db = with_summary(va,encode(net,validation)),with_summary(tr,encode(net,train))
            pool = candidates(q,db,128,None,max_work=True)
            coverage = float((pool[:,:,None] == valtruth[:,None]).any(1).float().mean())
        row = dict(epoch=epoch,validation_coverage=coverage,loss=float(np.mean(losses)) if losses else None)
        history.append(row)
        state = dict(epoch=epoch,state_dict=net.state_dict(),optimizer=optimizer.state_dict(),protocol=protocol)
        torch.save(state,args.output/'latest.pt')
        if coverage>best:
            best=coverage
            torch.save(state,args.output/'best.pt')
        if epoch==args.epochs:
            torch.save(state,args.output/'final.pt')
        save(args.output/'history.json',history)
        print(json.dumps(row),flush=True)
    (args.output/'ALL_DONE').write_text('completed\n')


if __name__=='__main__':
    main()
