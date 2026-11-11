"""Same DFD exclusion objective, using each city's own DFD train labels."""
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from external_city_train import ROOT,TASK,OPS,CODE,SDD,unpack
sys.path.insert(0,str(CODE))
# The archived runner requires sdd_compat on the caller's PYTHONPATH.
from germany_train import save,sha
from germany_evaluate import quality


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--city',required=True);args=ap.parse_args()
    from leap import LEAP
    from leap import RetrievalConfig
    from leap.common.dual_field import fields_packed
    from sdd_compat import install_raw_compat
    raw_distance=install_raw_compat('dfd')
    torch.set_num_threads(4);torch.manual_seed(2000);np.random.seed(2000)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=SDD/args.city;z=np.load(folder/'raw_inputs.npz')
    paths=[unpack(z,'train'),unpack(z,'validation')]
    asset=json.loads((OPS/'sdd_probe_training'/args.city/'final_asset.json').read_text())
    source=np.asarray(asset['anchors'],np.float32)
    mean,std=np.asarray(asset['geometry_mean']),np.asarray(asset['geometry_std'])
    def make(theta): return LEAP(theta,mean,std,config=RetrievalConfig('dfd',dfd_initial='light'))
    index=make(source).encode(paths[0]);n=len(paths[0])
    lens=index.offsets[1:]-index.offsets[:-1]
    width=int(lens.max());j=torch.arange(width,device='cuda')[None]
    padded=index.points[(index.offsets[:-1,None]+j).clamp_max(len(index.points)-1)]
    ends=torch.stack([index.points[index.offsets[:-1]],index.points[index.offsets[1:]-1]],1)
    teacher=torch.tensor(np.load(folder/'dfd_train.npy')/std[0],dtype=torch.float32,device='cuda')
    teacher.fill_diagonal_(torch.inf)
    positive=teacher.argsort(dim=1,stable=True)[:,:20]
    radius=teacher.gather(1,positive)[:,-1]
    val=np.load(folder/'dfd_validation.npy').copy();np.fill_diagonal(val,np.inf)
    truth=np.argsort(val,axis=1,kind='stable')[:,:50]
    theta=torch.nn.Parameter(torch.tensor(source,device='cuda'))
    def vertex(ps,ls):
        result=[]
        for lo in range(0,len(ps),16):
            p=ps[lo:lo+16];length=ls[lo:lo+16];w=int(length.max())
            squared=(p[:,:w,None]-theta[None,None]).square().sum(-1)
            valid=torch.arange(w,device='cuda')[None]<length[:,None]
            result.append(squared.masked_fill(~valid[:,:,None],torch.inf).amin(1).clamp_min(1e-24).sqrt())
        return torch.cat(result)
    qi=torch.arange(16,device='cuda');di=(qi*17+31)%n
    exact=raw_distance(index,index,qi,di)
    torch.testing.assert_close(exact,teacher[qi,di],rtol=3e-5,atol=2e-5)
    torch.testing.assert_close(vertex(padded[:16],lens[:16]),index.coordinates[:16,:64],rtol=2e-5,atol=2e-6)
    out=OPS/'sdd_dfd_training'/args.city;out.mkdir(parents=True,exist_ok=False)
    protocol=dict(city=args.city,metric='dfd',seed=2000,epochs=10,train=n,validation=len(paths[1]),
        initializer='same-city trained Hausdorff coordinates; new optimizer; DFD-only loss thereafter',
        parameters=128,learning_rate=.002,batch=64,negative_ranks=[1,4,8,16,24,32,64,128],
        selection='fixed E10; E0 and per-epoch validation retained as controls',test_loaded=False,
        split_protocol_sha256=sha(folder/'protocol.json'),teacher_parity=True,field_parity=True)
    save(out/'protocol.json',protocol)
    optimizer=torch.optim.Adam([theta],lr=.002);rng=torch.Generator().manual_seed(2000)
    ranks=torch.tensor([0,3,7,15,23,31,63,127],device='cuda')
    low=index.points.amin(0);high=index.points.amax(0);history=[]
    def validate(epoch,loss):
        m=make(theta.detach().cpu().numpy());ix=m.encode(paths[1]);tree=m.build_search_index(ix)
        initial=m.candidate_indices(ix,search_index=tree)
        complete=m.search_range(ix,search_index=tree,query_batch=32)
        row=dict(epoch=epoch,loss=loss,initial=quality(initial.cpu().numpy(),truth),
                 complete=quality(complete['indices'].cpu().numpy(),truth),refined_pairs=int(complete['counts'].sum()))
        history.append(row);save(out/'history.json',history)
        a=theta.detach().cpu().numpy()
        result=dict(asset,anchors=a.tolist(),anchors_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                    supervision_distance='dfd',representation='vertex/segment fields with DFD endpoint and ordered bounds',
                    city=args.city,training_city=args.city,epoch=epoch,training=protocol)
        save(out/f'epoch_{epoch:03d}.json',result)
        if epoch==10: save(out/'final_asset.json',result)
        print(json.dumps(row),flush=True)
    validate(0,None)
    for epoch in range(1,11):
        losses=[]
        for rows in torch.randperm(n,generator=rng).cuda().split(64):
            b=len(rows)
            with torch.no_grad():
                z=fields_packed(index.points,index.offsets,theta)[:,:64].contiguous()
                score=torch.cdist(z[rows],z,p=float('inf'))
                endpoint=torch.maximum(torch.cdist(ends[rows,0],ends[:,0],compute_mode='donot_use_mm_for_euclid_dist'),
                    torch.cdist(ends[rows,1],ends[:,1],compute_mode='donot_use_mm_for_euclid_dist'))
                score=torch.maximum(score,endpoint)
                score[torch.arange(b,device='cuda'),rows]=torch.inf;score.scatter_(1,positive[rows],torch.inf)
                negatives=score.topk(128,largest=False).indices[:,ranks]
                chosen,inv=torch.unique(torch.cat([rows,negatives.flatten()]),return_inverse=True)
            optimizer.zero_grad()
            a=vertex(padded[chosen],lens[chosen]);qa,da=a[inv[:b]],a[inv[b:]].reshape(b,8,64)
            endpoint=torch.linalg.vector_norm(ends[rows,None]-ends[negatives],dim=-1).amax(-1)
            bound=torch.maximum((qa[:,None]-da).abs().amax(-1),endpoint)
            target=radius[rows,None]+.5*(teacher[rows[:,None],negatives]-radius[rows,None])
            loss=(target-bound).clamp_min(0).mean();loss.backward()
            assert torch.isfinite(loss) and theta.grad is not None and torch.isfinite(theta.grad).all()
            torch.nn.utils.clip_grad_norm_([theta],1.);optimizer.step()
            with torch.no_grad(): theta.clamp_(low,high)
            losses.append(float(loss))
        torch.save(dict(theta=theta.detach().cpu(),optimizer=optimizer.state_dict(),epoch=epoch,
                        shuffle_rng=rng.get_state(),protocol=protocol),out/'latest.pt')
        validate(epoch,float(np.mean(losses)))
    save(out/'ALL_DONE.json',dict(status='trained',epoch=10,time=time.time(),test_evaluation_pending=True))


if __name__=='__main__': main()
