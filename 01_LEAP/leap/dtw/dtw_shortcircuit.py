"""Exact target-first evaluation of the two existing DTW box-sum bounds."""
import triton
import triton.language as tl

from leap.dtw.dtw_range import box_distance


@triton.jit
def target_first(P,OFF,QP,QOFF,BOX,QBOX,row,ids,valid,threshold,
                 N:tl.constexpr,TILE:tl.constexpr):
    j=tl.arange(0,N)
    xmin=tl.load(QBOX+row*4);ymin=tl.load(QBOX+row*4+1)
    xmax=tl.load(QBOX+row*4+2);ymax=tl.load(QBOX+row*4+3)
    d0=tl.load(OFF+ids,valid,other=0)
    nd=tl.load(OFF+ids+1,valid,other=0)-d0
    u=tl.load(P+(d0[:,None]+j[None,:])*2,valid[:,None]&(j[None,:]<nd[:,None]),other=0.)
    v=tl.load(P+(d0[:,None]+j[None,:])*2+1,valid[:,None]&(j[None,:]<nd[:,None]),other=0.)
    first=tl.sum(tl.where(j[None,:]<nd[:,None],box_distance(u,v,xmin,ymin,xmax,ymax),0.),1)
    alive=valid&(first<=threshold)
    second_checks=tl.full((),0,tl.int64)
    second_slots=tl.full((),0,tl.int64)
    # The max exceeds the radius as soon as either existing direction does.
    if tl.sum(alive.to(tl.int32),0)>0:
        lo=tl.load(QOFF+row);nq=tl.load(QOFF+row+1)-lo
        x=tl.load(QP+(lo+j)*2,j<nq,other=0.)
        y=tl.load(QP+(lo+j)*2+1,j<nq,other=0.)
        bx0=tl.load(BOX+ids*4,alive,other=0.);by0=tl.load(BOX+ids*4+1,alive,other=0.)
        bx1=tl.load(BOX+ids*4+2,alive,other=0.);by1=tl.load(BOX+ids*4+3,alive,other=0.)
        distance=box_distance(x[None,:],y[None,:],bx0[:,None],by0[:,None],bx1[:,None],by1[:,None])
        second=tl.sum(tl.where(j[None,:]<nq,distance,0.),1)
        second_checks=tl.sum(alive.to(tl.int64),0)*nq
        second_slots=tl.full((),N*TILE,tl.int64)
        alive=alive&(second<=threshold)
    first_checks=tl.sum(tl.where(valid,nd,0).to(tl.int64),0)
    return alive,first_checks,second_checks,second_slots
