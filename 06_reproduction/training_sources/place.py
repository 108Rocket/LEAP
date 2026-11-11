"""Train-only discrete probe placement; formal model and labels remain read-only."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import triton
import triton.language as tl
from leap.common.dual_field import DualFieldRetriever, fields_packed, scan
from leap.training.data import load_raw_splits
from leap.common.data import load_distances
from leap.hausdorff.range_search import pair_values
from diagnostic_kernel import pair_stats

# Supplied by leap.reproduction.training.paper before placement starts.
ROOT = RAW = TEACHER = None
NEG = 128


@triton.jit
def covers_kernel(Z, A, Q, D, W, H, TARGET, EXACT, BALL, P:tl.constexpr, M:tl.constexpr, B:tl.constexpr):
    i = tl.program_id(0)*B+tl.arange(0,B)
    pair = i//M; j = i%M
    q = tl.load(Q+pair,pair<P,other=0); d = tl.load(D+pair,pair<P,other=0)
    qa = tl.load(Z+q*2*M+j,pair<P,other=0); qb = tl.load(Z+q*2*M+M+j,pair<P,other=0)
    da = tl.load(Z+d*2*M+j,pair<P,other=0); db = tl.load(Z+d*2*M+M+j,pair<P,other=0)
    score = tl.maximum(tl.maximum(qb-da,db-qa),0.)
    target = tl.load(TARGET+pair,pair<P,other=0)
    margin = 2e-5*(1+tl.abs(target))
    exact = score>target+margin
    x = tl.load(A+j*2)-tl.load(W+pair*2,pair<P,other=0)
    y = tl.load(A+j*2+1)-tl.load(W+pair*2+1,pair<P,other=0)
    h = tl.load(H+pair,pair<P,other=0)
    sufficient = h-2*tl.sqrt(x*x+y*y)>target+2*margin
    tl.store(EXACT+i,exact,pair<P)
    tl.store(BALL+i,sufficient,pair<P)


def coverage_matrix(z,anchors,q,d,w,h,target):
    p,m = len(q),len(anchors)
    exact = torch.empty((p,m),dtype=torch.bool,device='cuda')
    ball = torch.empty_like(exact)
    covers_kernel[(triton.cdiv(p*m,256),)](z,anchors,q,d,w,h,target,exact,ball,p,m,256,
                                         num_warps=4,enable_fp_fusion=False)
    assert not (ball & ~exact).any(), 'Sufficient disk claimed exclusion without actual margin'
    return exact,ball


def choose(matrix, count, initial=None, allowed=None, quota=None):
    n,m = matrix.shape
    current = torch.zeros(n,dtype=torch.bool,device='cuda') if initial is None else initial.clone()
    available = torch.ones(m,dtype=torch.bool,device='cuda') if allowed is None else allowed.clone()
    ids,trace = [],[]
    for _ in range(count):
        hits = (matrix & ~current[:,None]).view(7000,NEG,m).sum(1,dtype=torch.int32)
        gain = hits.sum(0)
        if quota is not None:
            needed = (quota-current.view(7000,NEG).sum(1)).clamp_min(0)
            capped = torch.minimum(hits,needed[:,None]).sum(0)
            # Primary capped query coverage, secondary unsaturated pair coverage, then ID.
            primary = capped.masked_fill(~available,-1)
            tied = available & (primary==primary.max())
            gain = gain.masked_fill(~tied,-1)
        else:
            gain = gain.masked_fill(~available,-1)
        pick = int(gain.argmax())
        assert available[pick]
        available[pick] = False; current |= matrix[:,pick]; ids.append(pick)
        trace.append({'step':len(ids),'id':pick,'covered_pairs':int(current.sum()),
                      'queries_meeting_quota':int((current.view(7000,NEG).sum(1)>=98).sum())})
    return ids,trace


def save_asset(name,anchors,model,extra):
    a = anchors.cpu().numpy().astype(np.float32)
    payload = {'anchors':a.tolist(),'anchors_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
               'geometry_mean':model.mean.tolist(),'geometry_std':model.std.tolist(),
               'model':'dual-field','source_task':ROOT.name,'test_checkpoint_selection':False,
               'placement':extra}
    (ROOT/'assets'/f'{name}.json').write_text(json.dumps(payload,indent=2))


def cpu_h(a,b):
    def directed(x,y):
        edges = y[1:]-y[:-1]; delta = x[:,None]-y[None,:-1]
        u = np.clip((delta*edges[None]).sum(-1)/np.maximum((edges*edges).sum(-1),1e-30)[None],0,1)
        return np.sqrt(((delta-u[...,None]*edges[None])**2).sum(-1).min(1).max())
    return max(directed(a,b),directed(b,a))


@torch.no_grad()
def main():
    if any(path is None for path in (ROOT, RAW, TEACHER)):
        raise ValueError("Use python -m leap train --config <file> --recipe paper-probes")
    ROOT.mkdir(parents=True,exist_ok=True); (ROOT/'assets').mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(2000)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    model = DualFieldRetriever.from_asset()
    source = {'geometry_mean':model.mean,'geometry_std':model.std}
    splits = load_raw_splits(RAW,source)
    assert list(map(len,splits))==[7000,1000,2000]
    # Only train geometry and labels are used below, never validation/test distances.
    d = model.encode(splits[0]); labels = load_distances(TEACHER)
    teacher = labels[0].cuda().clone()*float(labels[3]); teacher.fill_diagonal_(torch.inf)
    order = teacher.argsort(dim=1,stable=True)
    positives = order[:,:20]; radius = teacher.gather(1,positives)[:,-1]
    qlist,dlist = [],[]
    for lo in range(0,7000,128):
        hi = min(lo+128,7000); q = torch.arange(lo,hi,device='cuda')
        values = scan(d.coordinates[lo:hi],d.coordinates)
        values[torch.arange(hi-lo,device='cuda'),q] = torch.inf
        values.scatter_(1,positives[lo:hi],torch.inf)
        neg = values.argsort(dim=1,stable=True)[:,:NEG]
        qlist.append(q[:,None].expand_as(neg).reshape(-1)); dlist.append(neg.reshape(-1))
    q,db = torch.cat(qlist),torch.cat(dlist)
    h,wids = pair_stats(d,torch.zeros(len(d.points),dtype=torch.int32,device='cuda'),q,db)
    h = h[:,0].contiguous(); w = d.points[wids.long()].contiguous()
    teacher_error = float((h-teacher[q,db]).abs().max())
    worst = int((h-teacher[q,db]).abs().argmax()); qi,di = int(q[worst]),int(db[worst])
    original = [(np.asarray(splits[0][i],np.float64)-model.mean)/model.std for i in [qi,di]]
    runtime = [d.points[int(d.offsets[i]):int(d.offsets[i+1])].cpu().numpy().astype(np.float64) for i in [qi,di]]
    cpu_original,cpu_runtime = cpu_h(*original),cpu_h(*runtime)
    precision_audit = {'worst_pair':[qi,di],'gpu_raw':float(h[worst]),'stored_teacher':float(teacher[qi,di]),
        'cpu_original_fp64':cpu_original,'cpu_encoded_fp64':cpu_runtime,'input_dtype':str(splits[0][qi].dtype)}
    coordinate_rounding_budget = sum(float(np.linalg.norm(np.spacing(np.abs(splits[0][i])).astype(np.float64)*.5/model.std,axis=1).max()) for i in [qi,di])
    precision_audit['pair_coordinate_rounding_budget'] = coordinate_rounding_budget
    (ROOT/'precision_audit.json').write_text(json.dumps(precision_audit,indent=2))
    print(json.dumps({'precision_audit':precision_audit}),flush=True)
    assert abs(float(h[worst])-cpu_runtime)<1e-5
    assert abs(float(teacher[qi,di])-cpu_original)<coordinate_rounding_budget+2e-5
    # Preserve official positive IDs; make their radius consistent with runtime geometry.
    positive_rows = torch.arange(7000,device='cuda')[:,None].expand_as(positives).reshape(-1)
    radius = pair_values(d,d,positive_rows,positives.reshape(-1)).reshape(7000,20).amax(1)
    target = (radius[q]+.5*(h-radius[q])).contiguous()
    old_exact,old_ball = coverage_matrix(d.coordinates,model.anchors,q,db,w,h,target)
    keep,protect_trace = choose(old_exact,48)
    initial = old_exact[:,keep].any(1)
    # Query-stratified training witnesses; no validation/test-driven pool expansion.
    rng = np.random.default_rng(2000)
    query_order = rng.permutation(7000)
    current = old_exact.any(1).view(7000,NEG).cpu().numpy()
    pool,seen = [],set()
    witness_cpu = w.cpu().numpy().reshape(7000,NEG,2)
    for qi in query_order:
        options = np.flatnonzero(~current[qi])
        if len(options)==0:
            options = np.arange(NEG)
        point = witness_cpu[qi,rng.choice(options)]
        key = point.tobytes()
        if key not in seen:
            seen.add(key); pool.append(point.copy())
        if len(pool)==1024: break
    assert len(pool)==1024
    anchors = torch.cat((model.anchors,torch.from_numpy(np.stack(pool)).cuda()))
    z = fields_packed(d.points,d.offsets,anchors)
    exact,ball = coverage_matrix(z,anchors,q,db,w,h,target)
    assert torch.equal(exact[:,:64],old_exact)
    # Protect already-established exact exclusions; new sites use disk guarantees in this arm.
    ball[:,:64] = exact[:,:64]
    available = torch.ones(len(anchors),device='cuda',dtype=torch.bool); available[keep]=False
    picked_ball,ball_trace = choose(ball,16,initial,available,quota=NEG-(50-20))
    picked_exact,exact_trace = choose(exact,16,initial,available,quota=NEG-(50-20))
    random_ids = (rng.choice(1024,16,replace=False)+64).tolist()
    arms = {'baseline':list(range(64)),'random':keep+random_ids,
            'region':keep+picked_ball,'exact_region':keep+picked_exact}
    train_stats = {}
    for name,ids in arms.items():
        assert len(ids)==64 and len(set(ids))==64
        covered = exact[:,ids].any(1).view(7000,NEG)
        train_stats[name] = {'exact_margin_coverage':float(covered.float().mean()),
            'queries_meeting_mined_quota':int((covered.sum(1)>=98).sum()),
            'new_sites':sum(i>=64 for i in ids)}
        save_asset(name,anchors[ids],model,{'method':name,'candidate_ids':ids,'training_only':True})
    # Numerical witness identity audit: putting a probe at w recovers H for sampled pairs.
    sample = torch.arange(0,len(q),max(1,len(q)//64),device='cuda')[:64]
    wz = fields_packed(d.points,d.offsets,w[sample])
    qa,qb = wz[q[sample]].chunk(2,-1); da,dbb = wz[db[sample]].chunk(2,-1)
    witness_score = torch.maximum(qb-da,dbb-qa).diagonal().clamp_min(0)
    err = float((witness_score-h[sample]).abs().max()); assert err<1e-5,err
    protocol = {'task':ROOT.name,'seed':2000,'source_asset':model.asset_id,'training_rows':7000,
        'test_used_for_placement':False,'validation_used_for_placement':False,'gradient_training':False,
        'candidates_new':1024,'protected':48,'replacement_budget':16,'negatives_per_query':NEG,
        'target':'r20 + 0.5*(H-r20), same margin shape as prior training',
        'radius_geometry':'Maximum runtime raw distance over official train Top20 IDs; labels are not rewritten',
        'disk':'H - 2||p-w|| > target + numerical slack; sufficient, not necessary',
        'selection':'capped per-query coverage98/128 then residual count then stable candidate ID',
        'quota_caveat':'Mined pool only; not a full-database recall guarantee',
        'steps':'No post-placement fine-tuning: isolate discrete position choice first',
        'teacher_max_error':teacher_error,'witness_identity_max_error':err,
        'precision_audit':precision_audit,
        'disk_implies_exact_checked_pairs_sites':len(q)*len(anchors),
        'training_stats':train_stats,'seconds':time.perf_counter()-start,
        'protected_ids':keep,'traces':{'protection':protect_trace,'region':ball_trace,'exact':exact_trace}}
    (ROOT/'placement.json').write_text(json.dumps(protocol,indent=2))
    np.savez_compressed(ROOT/'placement_inputs.npz',queries=q.cpu().numpy(),databases=db.cpu().numpy(),
        witnesses=w.cpu().numpy(),raw_h=h.cpu().numpy(),target=target.cpu().numpy(),candidate_sites=anchors.cpu().numpy())
    print(json.dumps(protocol),flush=True)
    (ROOT/'PLACEMENT_DONE').write_text('complete\n')


if __name__=='__main__': main()
