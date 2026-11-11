"""Protect complementary exclusion margins using training pairs only."""
import hashlib
import json
import numpy as np
import torch
from config import ROOT,RAW,TEACHER,START,setup
from learned_fields import LearnedFields
from leap.common.model import load_probe_asset
from leap.training.data import load_raw_splits
from leap.common.data import load_distances
from leap.common.dual_field import scan


@torch.no_grad()
def main():
    setup()
    asset=load_probe_asset();warm=torch.load(START,weights_only=False,map_location='cpu')
    assert warm['epoch']==50
    anchors=warm['state']['anchors'].numpy()
    model=LearnedFields(anchors,asset['geometry_mean'],asset['geometry_std'],'radial')
    train=load_raw_splits(RAW,asset)[0]
    labels=load_distances(TEACHER)
    teacher=labels[0].cuda().clone();teacher.fill_diagonal_(torch.inf)
    maximum=float(labels[3]);nearest=teacher.topk(20,largest=False)
    truth=nearest.indices;radius=nearest.values[:,-1]*maximum
    index=model.encode(train);z=index.coordinates
    valid_features=[];valid_targets=[];query_ids=[];database_ids=[]
    total=0
    for start in range(0,7000,128):
        q=torch.arange(start,min(start+128,7000),device='cuda')
        scores=scan(z[q],z)
        scores[torch.arange(len(q),device='cuda'),q]=torch.inf
        scores.scatter_(1,truth[q],torch.inf)
        negatives=scores.topk(64,largest=False).indices
        target=radius[q,None]+.5*(teacher[q[:,None],negatives]*maximum-radius[q,None])
        qa,qb=z[q].chunk(2,-1);da,db=z[negatives].chunk(2,-1)
        features=torch.maximum(qb[:,None]-da,db-qa[:,None]).clamp_min(0)
        protected=features.max(-1).values>=target
        valid_features.append(features[protected]);valid_targets.append(target[protected])
        query_ids.append(q[:,None].expand_as(negatives)[protected]);database_ids.append(negatives[protected])
        total+=negatives.numel()
    features=torch.cat(valid_features);targets=torch.cat(valid_targets)
    assert len(features)>0
    # Greedily maximize current margins up to their required threshold. Once
    # saturated, prefer complementary residual score support, then stable ID.
    capped=torch.minimum(features/targets[:,None].clamp_min(1e-12),torch.ones_like(features))
    current=torch.zeros(len(features),device='cuda')
    raw_current=current.clone();taken=torch.zeros(64,device='cuda',dtype=torch.bool)
    selected=[];trace=[]
    for _ in range(48):
        gain=(capped-current[:,None]).clamp_min(0).sum(0)
        gain[taken]=-torch.inf
        if float(gain.max())<=1e-6:
            gain=(features-raw_current[:,None]).clamp_min(0).sum(0);gain[taken]=-torch.inf
        choice=int(gain.argmax());selected.append(choice);taken[choice]=True
        current=torch.maximum(current,capped[:,choice]);raw_current=torch.maximum(raw_current,features[:,choice])
        trace.append({'count':len(selected),'chosen':choice,'retained_margin_fraction':float((raw_current>=targets).float().mean())})
    random_ids=np.random.default_rng(2000).choice(64,48,replace=False).tolist()
    def stats(ids):
        keep=features[:,ids].max(1).values
        return {'retained_margin_fraction':float((keep>=targets).float().mean()),
                'mean_lost_score':float((features.max(1).values-keep).mean())}
    result={'protect48':selected,'random48':random_ids,
            'source':'original E50 checkpoint and7000 training trajectories/labels only',
            'source_anchor_sha256':hashlib.sha256(anchors.tobytes()).hexdigest(),
            'mined_pairs':total,'already_meeting_margin_pairs':len(features),
            'selected_stats':stats(selected),'random_stats':stats(random_ids),'greedy_trace':trace}
    (ROOT/'selection.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(ROOT/'selection_pairs.npz',query=torch.cat(query_ids).cpu().numpy(),
                        database=torch.cat(database_ids).cpu().numpy(),target=targets.cpu().numpy())
    print(json.dumps({k:v for k,v in result.items() if k!='greedy_trace'},indent=2),flush=True)


if __name__=='__main__':main()
