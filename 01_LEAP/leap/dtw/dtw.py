"""Unconstrained DTW: sum of Euclidean costs on all original ordered vertices."""
import torch
import triton
import triton.language as tl


@triton.jit
def dtw_kernel(P, OFF, T, TOFF, Q, D, OUT, WORK, BLOCK: tl.constexpr):
    pair = tl.program_id(0)
    q = tl.load(Q + pair)
    d = tl.load(D + pair)
    a0 = tl.load(OFF + q)
    na = tl.load(OFF + q + 1) - a0
    b0 = tl.load(TOFF + d)
    nb = tl.load(TOFF + d + 1) - b0
    j = tl.arange(0, BLOCK)
    bx = tl.load(T + (b0 + j) * 2, j < nb, other=0.)
    by = tl.load(T + (b0 + j) * 2 + 1, j < nb, other=0.)
    work = WORK + pair * 3 * BLOCK
    for bank in range(3):
        tl.store(work + bank * BLOCK + j, float('inf'))
    tl.debug_barrier()
    for diagonal in range(na + nb - 1):
        i = diagonal - j
        valid = (i >= 0) & (i < na) & (j < nb)
        ax = tl.load(P + (a0 + i) * 2, valid, other=0.)
        ay = tl.load(P + (a0 + i) * 2 + 1, valid, other=0.)
        previous_bank = (diagonal + 2) % 3
        previous2_bank = (diagonal + 1) % 3
        previous = tl.load(work + previous_bank * BLOCK + j, volatile=True)
        left = tl.load(work + previous_bank * BLOCK + j - 1,
                       j > 0, other=float('inf'), volatile=True)
        cross = tl.load(work + previous2_bank * BLOCK + j - 1,
                        j > 0, other=float('inf'), volatile=True)
        predecessor = tl.minimum(previous, tl.minimum(left, cross))
        predecessor = tl.where((i == 0) & (j == 0), 0., predecessor)
        cost = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
        current = tl.where(valid, tl.sqrt(cost) + predecessor, float('inf'))
        tl.store(work + (diagonal % 3) * BLOCK + j, current)
        tl.debug_barrier()
    value = tl.load(work + ((na + nb - 2) % 3) * BLOCK + nb - 1, volatile=True)
    tl.store(OUT + pair, value)


@torch.no_grad()
def dtw_pairs(queries, database, qids, dids):
    if qids.shape != dids.shape or qids.ndim != 1:
        raise ValueError('Expected one-dimensional matching pair ID arrays')
    if qids.dtype != torch.int64 or dids.dtype != torch.int64:
        raise ValueError('Pair IDs must be int64')
    if any(t.device != queries.points.device for t in (database.points, qids, dids)):
        raise ValueError('Pairs and geometry must share a CUDA device')
    out = torch.empty(dids.shape, device=dids.device, dtype=torch.float32)
    if not dids.numel():
        return out
    maximum = int((database.offsets[1:] - database.offsets[:-1]).max())
    block = triton.next_power_of_2(maximum)
    # Bound temporary rolling buffers; chunk pairs, never truncate source paths.
    chunk = max(1, min(dids.numel(), (32 * 1024 * 1024) // (3 * block)))
    work = torch.empty((chunk, 3, block), device=dids.device, dtype=torch.float32)
    qids, dids = qids.contiguous(), dids.contiguous()
    for start in range(0, dids.numel(), chunk):
        stop = min(start + chunk, dids.numel())
        dtw_kernel[(stop - start,)](queries.points, queries.offsets,
            database.points, database.offsets, qids[start:stop], dids[start:stop],
            out[start:stop], work, block, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def pse_kernel(QZ,DZ,QP,QO,DP,DO,SELF,OUT,NQ:tl.constexpr,ND:tl.constexpr,
               QSTART,DSTART,SAME:tl.constexpr):
    q=tl.program_id(0)
    j=tl.program_id(1)*16+tl.arange(0,16)
    f=tl.arange(0,64)
    qa=tl.load(QZ+(q+QSTART)*128+f)
    da=tl.load(DZ+(j[:,None]+DSTART)*128+f[None,:],j[:,None]<ND,other=0.)
    score=tl.max(tl.abs(qa[None,:]-da),1)
    for end in range(2):
        qi=tl.load(QO+q+QSTART+end)-end
        di=tl.load(DO+j+DSTART+end,j<ND,other=0)-end
        dx=tl.load(QP+qi*2)-tl.load(DP+di*2,j<ND,other=0.)
        dy=tl.load(QP+qi*2+1)-tl.load(DP+di*2+1,j<ND,other=0.)
        score=tl.maximum(score,tl.sqrt(dx*dx+dy*dy))
    if SAME:
        own=tl.load(SELF+q+QSTART)
        score=tl.where(j+DSTART==own,float('inf'),score)
    tl.store(OUT+q*ND+j,score,j<ND)


@torch.no_grad()
def pse_candidates(queries,database,k,qids=None):
    """LB PSE score, blocked full scan; qids identifies excluded database IDs."""
    from leap.common.interval_index import lex_keys
    nq,nd=len(queries.coordinates),len(database.coordinates)
    if not 1<=k<=nd:
        raise ValueError('Invalid PSE candidate count')
    result=[]
    for start in range(0,nq,2000):
        stop=min(start+2000,nq)
        best=None
        for dstart in range(0,nd,4096):
            width=min(4096,nd-dstart)
            score=torch.empty((stop-start,width),device=queries.points.device,dtype=torch.float32)
            pse_kernel[(stop-start,triton.cdiv(width,16))](queries.coordinates,database.coordinates,
                queries.points,queries.offsets,database.points,database.offsets,
                qids if qids is not None else queries.offsets,score,stop-start,width,start,dstart,qids is not None,
                num_warps=4,enable_fp_fusion=False)
            ids=torch.arange(dstart,dstart+width,device=score.device)[None].expand_as(score)
            keys=lex_keys(score,ids).topk(min(k,width),largest=False).values
            if best is not None:
                keys=torch.cat((best,keys),1)
                keys=keys.topk(min(k,keys.shape[1]),largest=False).values
            best=keys
        result.append(best&0xffffffff)
    if not result:
        return torch.empty((0,k),device=queries.points.device,dtype=torch.int64)
    return torch.cat(result)
