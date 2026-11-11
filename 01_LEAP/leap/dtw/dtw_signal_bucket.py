"""Length bucketing for ordered-probe alignment (eq:dtw-signal).

All target vertices remain present. Each target receives the smallest power-
of-two padded block that holds its length, instead of the database maximum.
"""
import torch
import triton
from leap.dtw.dtw_signal import signal_kernel, encode


@torch.no_grad()
def pairs(q,db,qids,dids,theta=None,fields=None,chunk=16384,*,batch_plan=None):
    if batch_plan is not None:
        return batch_plan.pairs(q,db,qids,dids,theta=theta,fields=fields,chunk=chunk)
    from leap.common.runtime.context import current
    plan = current()
    if plan is not None and fields is not None:
        return plan.signal_values(q, db, qids, dids, fields)
    a,b=(encode(q,theta),encode(db,theta)) if fields is None else fields
    out=torch.empty(dids.shape,device=dids.device,dtype=torch.float32)
    if not len(dids):
        return out
    lengths=db.offsets[dids+1]-db.offsets[dids]
    maximum=triton.next_power_of_2(int((db.offsets[1:]-db.offsets[:-1]).max()))
    lower=0
    for block in [2**i for i in range(maximum.bit_length())]:
        selected=((lengths>lower)&(lengths<=block)).nonzero().flatten()
        lower=block
        if not len(selected):
            continue
        qs=qids[selected].contiguous();ds=dids[selected].contiguous()
        values=torch.empty(len(selected),device=dids.device,dtype=torch.float32)
        work=torch.empty((min(chunk,len(selected)),3,block),device=dids.device)
        for start in range(0,len(selected),chunk):
            end=min(start+chunk,len(selected))
            signal_kernel[(end-start,)](a,q.offsets,b,db.offsets,qs[start:end],ds[start:end],
                values[start:end],work,work,a.shape[1],block,False,num_warps=4,enable_fp_fusion=False)
        out[selected]=values
    return out
