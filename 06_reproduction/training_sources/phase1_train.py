import argparse
import hashlib
import json
import time
import numpy as np
import torch
from config import ROOT,RAW,TEACHER,NEGATIVE_RANKS,CONDITIONS,WITNESS_WEIGHTS,setup
from learned_fields import LearnedFields
from leap.common.model import load_probe_asset
from leap.training.data import load_raw_splits
from leap.common.data import load_distances, official_hitting_ratio
from leap.common.dual_field import scan, score_torch
from leap.common.raw import fused_raw_scores
from diagnostic_kernel import pair_stats


@torch.no_grad()
def validation(model,paths,target,maximum):
    idx=model.encode(paths)
    candidates=scan(idx.coordinates,idx.coordinates,True).topk(50,largest=False).indices.contiguous()
    values,order=fused_raw_scores(idx,idx,candidates).topk(20,largest=False)
    ids=candidates.gather(1,order)
    predicted=torch.full_like(target,torch.inf)
    predicted.scatter_(1,ids,values/maximum);predicted.fill_diagonal_(0)
    metrics={n:official_hitting_ratio(predicted,target,k,t) for n,k,t in [('hr5',5,5),('hr20',20,20),('r5_at20',20,5)]}
    truth=target.clone();truth.fill_diagonal_(torch.inf)
    gt=truth.topk(20,largest=False).indices
    covered=(gt[:,:,None]==candidates[:,None,:]).any(-1)
    metrics.update(coverage_gt20_at50=float(covered.double().mean()),missing_gt20=int((~covered).sum()))
    return metrics


def main(mode,resume=False):
    setup();torch.manual_seed(2000);np.random.seed(2000)
    out=ROOT/'training'/mode;out.mkdir(parents=True,exist_ok=resume)
    asset=load_probe_asset()
    initial=np.asarray(asset['initial_probes'],np.float32)[:,1].copy()
    splits=load_raw_splits(RAW,asset);labels=load_distances(TEACHER)
    assert [len(p) for p in splits]==[7000,1000,2000]
    model=LearnedFields(initial,asset['geometry_mean'],asset['geometry_std'],'radial')
    witness_weight=WITNESS_WEIGHTS[mode]
    index=model.encode(splits[0]);points,offsets=index.points,index.offsets
    unused_selection=torch.zeros(len(points),device='cuda',dtype=torch.int32)
    lengths=offsets[1:]-offsets[:-1]
    padded=points.new_zeros((len(lengths),int(lengths.max()),2))
    boundaries=offsets.cpu().tolist()
    for i,(a,b) in enumerate(zip(boundaries[:-1],boundaries[1:])):padded[i,:b-a]=points[a:b]
    maximum=float(labels[3]);teacher=labels[0].cuda().clone();teacher.fill_diagonal_(torch.inf)
    neighbors=teacher.topk(20,largest=False);truth=neighbors.indices;radius=neighbors.values[:,-1]*maximum
    target_val=labels[1].cuda();ranks=torch.tensor(NEGATIVE_RANKS,device='cuda')-1
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(2000)
    low,high=points.amin(0),points.amax(0)
    protocol={'mode':mode,'epochs':50,'steps':5500,'batch':64,'optimizer':'Adam','lr':.002,'clip':1.,'seed':2000,
              'initial_anchors_sha256':hashlib.sha256(initial.tobytes()).hexdigest(),
              'parameter_count':sum(p.numel() for p in model.parameters()),'features':128,'candidate_k':50,
              'positive_k':20,'budget_negative_ranks':NEGATIVE_RANKS,'budget_target':'r20+0.5*(teacher_negative-r20)',
              'pair_reduction':'mean across8 negatives and batch','witness_loss_weight':witness_weight,
              'witness_loss':'mean(active_base_deficit * relu(2*min_anchor_distance_to_raw_witness + target - raw_h))',
              'witness_source':'current train-only hard-negative pairs, exact raw geometry, detached',
              'precision':'FP32 TF32off','checkpoint_selection':'none; fixed epochs0/1/2/5/10/20/50',
              'test_selection_enabled':False,'test_used_for_gradients':False,
              'model_family':'unchanged radial dual fields;128 trainable anchor coordinates'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    history=[]
    first_epoch=1
    if resume:
        saved=torch.load(out/'latest.pt',weights_only=False)
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['shuffle_rng'])
        first_epoch=int(saved['epoch'])+1
        history=json.loads((out/'history.json').read_text())
        assert history[-1]['epoch']==saved['epoch']
        print('RESUME',mode,first_epoch,flush=True)
    def snapshot(epoch,stats):
        metrics=validation(model,splits[1],target_val,maximum)
        row={'mode':mode,'epoch':epoch,**stats,'validation':metrics}
        history.append(row)
        if epoch in (0,1,2,5,10,20,50):
            payload={'mode':mode,'epoch':epoch,'state':model.payload(),'mean':model.mean.tolist(),'std':model.std.tolist(),
                     'protocol':protocol,'validation':metrics}
            (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(payload,indent=2))
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
    if not resume:snapshot(0,{})
    for epoch in range(first_epoch,51):
        start_time=time.perf_counter();statistics=[]
        permutation=torch.randperm(7000,generator=generator).cuda()
        for start in range(0,7000,64):
            qids=permutation[start:start+64];b=len(qids)
            with torch.no_grad():
                allz=model.packed(points,offsets)
                coarse=scan(allz[qids],allz);coarse[torch.arange(b,device='cuda'),qids]=torch.inf
                nonpositive=coarse.clone();nonpositive.scatter_(1,truth[qids],torch.inf)
                neg=nonpositive.topk(31,largest=False).indices[:,ranks]
                targets=radius[qids,None]+.5*(teacher[qids[:,None],neg]*maximum-radius[qids,None])
                chosen,inverse=torch.unique(torch.cat((qids,neg.flatten())),return_inverse=True)
                if witness_weight:
                    exact,witness_id=pair_stats(index,unused_selection,qids[:,None].expand(-1,8).reshape(-1),neg.flatten())
                    raw_h=exact[:,0].reshape(b,8)
                    witness_points=points[witness_id.long()].reshape(b,8,2)
            optimizer.zero_grad(set_to_none=True)
            z=model(padded[chosen],lengths[chosen]);q=z[inverse[:b]];n=z[inverse[b:]].reshape(b,8,128)
            deficits=targets-score_torch(q[:,None],n)
            rank_loss=deficits.clamp_min(0).mean()
            witness_loss=rank_loss.new_zeros(())
            witness_distance=rank_loss.new_zeros(())
            if witness_weight:
                # For a raw Hausdorff witness w, s(c)>=h-2*||c-w||.
                # This sufficient-margin penalty assigns the closest anchor,
                # instead of only updating the current maximum-score anchor.
                distances=torch.linalg.vector_norm(model.anchors[None,None]-witness_points[:,:,None],dim=-1)
                nearest=distances.min(-1).values
                active=(deficits.detach()>0).float()
                witness_loss=((2*nearest+targets-raw_h).clamp_min(0)*active).mean()
                witness_distance=nearest.detach().mean()
            loss=rank_loss+witness_weight*witness_loss
            assert torch.isfinite(loss)
            loss.backward()
            bad=[name for name,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            if bad:
                diagnostic={'mode':mode,'epoch':epoch,'batch_start':start,'loss':float(loss),'bad_gradients':bad}
                if mode=='fourier':diagnostic['frequency_norm_min']=float(model.omega.norm(dim=-1).min())
                (out/'gradient_failure.json').write_text(json.dumps(diagnostic,indent=2))
                torch.save({'state':model.state_dict(),'chosen':chosen,'inverse':inverse,'targets':targets,'qids':qids},out/'gradient_failure.pt')
                raise FloatingPointError(json.dumps(diagnostic))
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();model.project(low,high)
            statistics.append([float(loss),float((deficits>0).float().mean()),float(grad),
                               float(rank_loss),float(witness_loss),float(witness_distance)])
        mean=np.mean(statistics,axis=0)
        stats=dict(zip(('loss','active_fraction','grad_norm','rank_loss','witness_loss','witness_distance'),mean))
        stats['train_seconds']=time.perf_counter()-start_time
        if mode=='ellipse':stats.update(axis_mean=float(model.axis.mean()),axis_min=float(model.axis.min()))
        if mode=='fourier':stats.update(amplitude_abs_mean=float(model.amplitude.abs().mean()),amplitude_abs_max=float(model.amplitude.abs().max()))
        torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'shuffle_rng':generator.get_state()},out/'latest.pt')
        snapshot(epoch,stats)
    (out/'TRAIN_DONE').write_text('Fixed50 epochs complete; no test selection.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=CONDITIONS)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();main(args.mode,args.resume)
