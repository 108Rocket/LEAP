"""City-local SDD Hausdorff supervision with fixed two-phase training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import time
import numpy as np
import torch
from external_city_train import ROOT,TASK,OPS,CODE,SDD,unpack
sys.path.insert(0,str(CODE))
from germany_train import save,sha


class Sizes(ast.NodeTransformer):
    def __init__(self,n,nv): self.n,self.nv=n,nv
    def visit_Assert(self,node):
        if ast.unparse(node.test)=='len(pool) == 1024':
            node.test=ast.parse('len(pool) >= 16',mode='eval').body
            return node
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and node.func.attr=='choice' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value==1024:
            node.args[0]=ast.parse('len(pool)',mode='eval').body
        return self.generic_visit(node)
    def visit_List(self,node):
        if all(isinstance(v,ast.Constant) for v in node.elts) and [v.value for v in node.elts]==[7000,1000,2000]:
            return ast.copy_location(ast.List(elts=[ast.Constant(v) for v in [self.n,self.nv,0]],ctx=ast.Load()),node)
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if type(node.value) is int and node.value==7000:
            return ast.copy_location(ast.Constant(self.n),node)
        if type(node.value) is int and node.value==1024:
            return ast.copy_location(ast.Constant(min(1024,self.n)),node)
        if type(node.value) is int and node.value==5500:
            return ast.copy_location(ast.Constant(50*((self.n+63)//64)),node)
        return node


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--city',required=True)
    ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args()
    assert torch.cuda.get_device_name()=='NVIDIA L20'
    import numba
    torch.set_num_threads(4);numba.set_num_threads(4)
    torch.manual_seed(2000);np.random.seed(2000)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=SDD/args.city
    z=np.load(data/'raw_inputs.npz')
    paths=[unpack(z,'train'),unpack(z,'validation'),[]]
    mean,std=z['geometry_mean'],z['geometry_std']
    tm=np.load(data/'haus_train.npy')/std[0]
    vm=np.load(data/'haus_validation.npy')/std[0]
    labels=(torch.tensor(tm,dtype=torch.float32),torch.tensor(vm,dtype=torch.float32),torch.empty((0,0)),1.0)
    n,nv=len(paths[0]),len(paths[1])
    first=int(tm.mean(1).argmin());ids=[first];near=tm[first].copy();near[first]=-np.inf
    while len(ids)<64:
        choice=int(near.argmax());ids.append(choice);near=np.minimum(near,tm[choice]);near[ids]=-np.inf
    norm=[((p-mean)/std).astype(np.float32) for p in paths[0]]
    probes=[]
    for i in ids:
        p=norm[i];edge=p[-1]-p[0]
        t=np.clip(((p-p[0])*edge).sum(1)/max(float((edge*edge).sum()),1e-30),0,1)
        residual=((p-p[0]-t[:,None]*edge)**2).sum(1)
        mid=int(residual[1:-1].argmax())+1
        probes.append(p[[0,mid,len(p)-1]])
    probes=np.asarray(probes,np.float32)
    initial=dict(initial_probes=probes.tolist(),geometry_mean=mean.tolist(),geometry_std=std.tolist())
    from leap.common import model
    from leap.training import data as evaluate
    from leap.common import data as geometry
    model.load_probe_asset=lambda path=None:initial
    evaluate.load_raw_splits=lambda path,asset:paths
    geometry.load_distances=lambda path:labels
    from sdd_point_fields import install,pair_stats
    install()
    source=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(source))
    import diagnostic_kernel
    diagnostic_kernel.pair_stats=pair_stats
    from learned_fields import LearnedFields
    net=LearnedFields(probes[:,1],mean,std,'radial')
    index=net.encode(paths[0])
    q=torch.arange(16,device='cuda');d=(q*17+31)%n
    raw,w=pair_stats(index,None,q,d)
    expected=labels[0].cuda()[q,d]
    torch.testing.assert_close(raw[:,0],expected,rtol=2e-5,atol=2e-5)
    lengths=index.offsets[1:]-index.offsets[:-1]
    padded=torch.zeros((16,int(lengths[:16].max()),2),device='cuda')
    for i in range(16):
        p=index.points[index.offsets[i]:index.offsets[i+1]]
        padded[i,:len(p)]=p
    value=net(padded,lengths[:16])
    torch.testing.assert_close(value,index.coordinates[:16],rtol=2e-5,atol=2e-6)
    value.sum().backward()
    assert torch.isfinite(net.anchors.grad).all() and net.anchors.grad.norm()>0
    out=OPS/'sdd_probe_training'/args.city
    if args.preflight_only:
        out=OPS/'sdd_probe_preflight'/(args.city+'.'+str(time.time_ns()))
    out.mkdir(parents=True,exist_ok=False)
    save(out/'PREFLIGHT.json',dict(city=args.city,train=n,validation=nv,teacher_parity=True,
        differentiable_packed_parity=True,nonzero_finite_gradient=True,initial_source_ids=ids,
        meaning='point-set SDD target: both stored fields use nearest original vertices',
        checkpoint_transfer=False,test_loaded=False))
    if args.preflight_only: return
    cfg=types.ModuleType('config')
    cfg.ROOT=out/'phase1';cfg.RAW=data/'raw_inputs.npz';cfg.TEACHER=data/'haus_train.npy'
    cfg.NEGATIVE_RANKS=(1,5,9,13,17,21,25,31)
    cfg.CONDITIONS=('baseline','protect48');cfg.WITNESS_WEIGHTS={'baseline':0.,'protect48':.1}
    cfg.setup=lambda:None
    sys.modules['config']=cfg
    from entrance.training import SeedLiterals
    def load(name,expected=None):
        path=source/(name+'.py')
        tree=Sizes(n,nv).visit(ast.parse(path.read_text()))
        tree=ast.fix_missing_locations(SeedLiterals(2000,expected).visit(tree))
        module=types.ModuleType('sdd_'+name);module.__file__=str(path)
        exec(compile(tree,str(path),'exec'),module.__dict__)
        return module
    del net,index,padded,value,raw
    torch.cuda.empty_cache()
    load('phase1_train').main('baseline')
    cfg.START=cfg.ROOT/'training/baseline/latest.pt'
    warm=torch.load(cfg.START,map_location='cpu',weights_only=False)
    expected=hashlib.sha256(warm['state']['anchors'].numpy().tobytes()).hexdigest()
    cfg.ROOT=out/'phase2';cfg.ROOT.mkdir()
    load('select_probes').main()
    load('phase2_train',expected).main('protect48')
    payload=json.loads((cfg.ROOT/'training/protect48/epoch_050.json').read_text())
    anchors=np.asarray(payload['state']['anchors'],np.float32)
    asset=dict(anchors=anchors.tolist(),anchors_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
        geometry_mean=mean.tolist(),geometry_std=std.tolist(),city=args.city,
        supervision_distance='SDD point-set Hausdorff',training_epochs=100)
    save(out/'protected_asset.json',asset)
    place=load('place')
    place.ROOT=out/'placement';place.RAW=cfg.RAW;place.TEACHER=cfg.TEACHER
    base=place.DualFieldRetriever
    place.DualFieldRetriever=types.SimpleNamespace(from_asset=lambda:base.from_asset(out/'protected_asset.json'))
    def cpu_h(a,b):
        c=np.linalg.norm(a[:,None]-b[None,:],axis=-1)
        return float(max(c.min(0).max(),c.min(1).max()))
    place.cpu_h=cpu_h
    place.main()
    final=json.loads((place.ROOT/'assets/exact_region.json').read_text())
    final.update(city=args.city,supervision_distance='SDD point-set Hausdorff',
        representation='point-set fields (a,a)',epochs=100,split_protocol_sha256=sha(data/'protocol.json'),
        test_selection_enabled=False,training_city=args.city)
    save(out/'final_asset.json',final)
    save(out/'ALL_DONE.json',dict(status='trained',asset=str(out/'final_asset.json'),test_evaluation_pending=True))


if __name__=='__main__': main()
