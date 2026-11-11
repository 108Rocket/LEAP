"""Input/checkpoint interfaces for unmodified author encoders."""
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

OPS=Path(__file__).resolve().parent
ROOT=OPS.parents[3]
SOURCES=Path(os.environ['LEAP_EXTERNAL_SOURCES']) if 'LEAP_EXTERNAL_SOURCES' in os.environ else None
EXPORT=Path(os.environ.get('LEAP_EXTERNAL_ASSETS', OPS/'assets'))
PRIOR=Path(os.environ.get('LEAP_EXTERNAL_PRIOR_ASSETS', EXPORT))
sys.path[:0]=[str(ROOT/'07_external_models/external_models'),str(ROOT/'07_external_models')]

def construct(model,metric):
    if model not in ('ConvTraj','SIMformer','TrajCL','MovSemCL','TSMini','TrajAgg'):
        raise ValueError('This publication excludes LB and supports only the six neural baselines')
    if metric not in ('hausdorff','discret_frechet','dtw'):
        raise ValueError('Use hausdorff, discret_frechet or dtw')
    from install_sources import verify_source
    source=verify_source(model,SOURCES)
    folder=(PRIOR if model in ('ConvTraj','SIMformer') else EXPORT)/f'Porto.{metric}.{model}'
    if model in ('ConvTraj','SIMformer'):
        sys.path.insert(0,str(source))
        spec=json.loads((folder/'inference.json').read_text())
        state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        if model=='SIMformer':
            from simformer.model import SIMformer
            net=SIMformer(2,128,256,16,1,'learnable').cuda().eval()
            mu,std=np.asarray(spec['mean']),np.asarray(spec['std'])
            def encode(rows):
                rows=[torch.as_tensor((p-mu)/std,dtype=torch.float32) for p in rows]
                lengths=torch.tensor([len(p) for p in rows],device='cuda')
                x=pad_sequence(rows,batch_first=True).cuda()
                return net(x,torch.arange(x.shape[1],device='cuda')[None]<lengths[:,None])
        else:
            from mynn.shortcut import ShortCutCNN
            from external_models.learned.ConvTraj.inputs import preprocess
            net=ShortCutCNN(128,128,128,128,20,200,8,torch.device('cuda:0'),1).cuda().eval()
            def encode(rows):
                seq,raster=preprocess(rows,spec['raster_spec'])
                seq,raster=seq.cuda(),raster.cuda()
                return net.encode(seq[:,:,0:1],seq[:,:,1:2],raster)
        net.load_state_dict(state['model'],strict=True)
        return encode,2,lambda p:np.asarray(p,np.float64)*111320.,folder/'best.pt'
    transform=lambda p:np.asarray(p,np.float64)*111320.
    sys.path.insert(0,str(source))
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    if model=='TSMini':
        from config import Config
        Config.dataset='porto'
        Config.post_value_updates()
        Config.device=torch.device('cuda:0')
        from model.tsmini import TSMini
        from task.trajsimi import collate_eval_test
        from utils.cellspace import create_cellspace
        stat=json.loads((folder/'statistics.json').read_text())
        lo,hi=stat['minimum'],stat['maximum']
        space=create_cellspace(lo[0],hi[0],lo[1],hi[1],Config.cell_size,Config.cell_size,Config.cellspace_buffer)
        net=TSMini().cuda().eval()
        net.load_state_dict(state['encoder'],strict=True)
        def encode(batch):
            x,lens=collate_eval_test([p.tolist() for p in batch],space,0)
            return net(x.cuda(),lens.cuda())
        return encode,1,transform,folder/'checkpoint.pt'
    if model=='TrajAgg':
        from utils.tools import merc2cell2
        from Model.AggTransformer import AggAttnEncoder
        from utils.cellspace import CellSpace
        stat=json.loads((folder/'statistics.json').read_text())
        mu,std=np.array(stat['mean']),np.array(stat['std'])
        lo,hi=np.array(stat['minimum'])-1000,np.array(stat['maximum'])+1000
        def lonlat(p):
            return np.degrees([p[0]/6378137.,2*np.arctan(np.exp(p[1]/6378137.))-np.pi/2])
        west,south=lonlat(lo)
        east,north=lonlat(hi)
        space=CellSpace(100,100,west,south,east,north)
        net=AggAttnEncoder(128,16,1,.5,.5,{'hausdorff':'haus','discret_frechet':'dfret','dtw':'dtw'}[metric]).cuda().eval()
        net.load_state_dict(state['model'],strict=True)
        def encode(batch):
            pairs=[merc2cell2(torch.as_tensor(p),space) for p in batch]
            rows=[torch.as_tensor((p.numpy()-mu)/std,dtype=torch.float32) for _,p in pairs]
            x=pad_sequence(rows,batch_first=True).cuda()
            grids=pad_sequence([g.float() for g,_ in pairs],batch_first=True).cuda()
            lengths=torch.tensor([len(p) for p in rows],device='cuda')
            return net(x,grids,None,torch.arange(x.shape[1],device='cuda')[None]<lengths[:,None])
        return encode,1 if metric=='dtw' else float('inf'),transform,folder/'checkpoint.pt'
    from config import Config
    Config.update(dict(dataset='porto',seed=2000))
    Config.device=torch.device('cuda:0')
    Config.dataset_prefix='sdd_Porto'
    Config.dumpfile_uniqueid='_city_local'
    Config.dataset_file=str(EXPORT/'city_grid/TRAINING_ONLY')
    Config.dataset_cell_file=str(EXPORT/'city_grid/cellspace.pkl')
    Config.dataset_embs_file=str(EXPORT/'city_grid/embeddings.pkl')
    Config.trajsimi_measure_fn_name=metric
    Config.trajsimi_encoder_name='TrajCL' if model=='TrajCL' else 'SemMovCL'
    Config.adaptive_mask_cache_file=str(EXPORT/'unused_inference_cache.pkl')
    module=importlib.import_module('model.trajcl' if model=='TrajCL' else 'model.semmovcl')
    net=(module.TrajCL() if model=='TrajCL' else module.SemMovCL()).cuda().eval()
    from task import trajsimi
    net.clmodel.encoder_q.load_state_dict(state['encoder_q'],strict=True)
    head=trajsimi.TrajSimiRegression(Config.seq_embedding_dim).cuda().eval()
    head.load_state_dict(state['trajsimi'],strict=True)
    with open(Config.dataset_cell_file,'rb') as f:
        space=pickle.load(f)
    with open(Config.dataset_embs_file,'rb') as f:
        embs=pickle.load(f).cuda()
    holder=SimpleNamespace(cellspace=space,cellembs=embs)
    def encode(batch):
        native=next(trajsimi.TrajSimi.trajsimi_dataset_generator_single_batchi(holder,[p.tolist() for p in batch]))
        return head(net.interpret(*native))
    return encode,1,transform,folder/'checkpoint.pt'
