"""Resume DTW's unchanged group bounds, with near-first short leaf waves."""
import triton
import triton.language as tl
from leap.dtw.dtw_range import box_distance
from leap.common.group_envelope_filter import node_bound_kernel,dtw_node_bound_kernel


@triton.jit
def leaf_resume(QP,QOFF,QZ,TREE,QBOX,TBOX,RADIUS,SLACK,RF,RT,LEAVES,COUNTS,VISITS,
                STACK,LOWER,SP,G:tl.constexpr,N:tl.constexpr,RADIAL:tl.constexpr,
                FIRST:tl.constexpr,LIMIT:tl.constexpr):
    row=tl.program_id(0)
    j=tl.arange(0,N);f=tl.arange(0,64);slot=tl.arange(0,32);sides=tl.arange(0,2)
    lo=tl.load(QOFF+row);nq=tl.load(QOFF+row+1)-lo
    x=tl.load(QP+(lo+j)*2,j<nq,other=0.)
    y=tl.load(QP+(lo+j)*2+1,j<nq,other=0.)
    aq=tl.load(QZ+row*128+f);bq=tl.load(QZ+row*128+64+f)
    threshold=tl.load(RADIUS+row)+tl.load(SLACK+row)
    if FIRST:
        stack=tl.where(slot==0,1,0);lbs=tl.full((32,),0.,tl.float32);sp=0
    else:
        sp=tl.load(SP+row)
        stack=tl.load(STACK+row*32+slot,slot<=sp,other=0)
        lbs=tl.load(LOWER+row*32+slot,slot<=sp,other=0.)
    count=tl.full((),0,tl.int32);nodes=tl.full((),0,tl.int32)
    active=sp>=0
    while active:
        node=tl.sum(tl.where(slot==sp,stack,0),0)
        lower=tl.sum(tl.where(slot==sp,lbs,0.),0);sp-=1
        if lower<=threshold:
            nodes+=1
            if node<G:
                child=node*2+sides
                cx0=tl.load(TBOX+child*4);cy0=tl.load(TBOX+child*4+1)
                cx1=tl.load(TBOX+child*4+2);cy1=tl.load(TBOX+child*4+3)
                distance=box_distance(x[None,:],y[None,:],cx0[:,None],cy0[:,None],cx1[:,None],cy1[:,None])
                boxsum=tl.sum(tl.where(j[None,:]<nq,distance,0.),1)
                ah=tl.load(TREE+child[:,None]*128+f[None,:])
                bl=tl.load(TREE+child[:,None]*128+64+f[None,:])
                bound=dtw_node_bound_kernel(node_bound_kernel(aq[None,:],bq[None,:],ah,bl,1),boxsum)
                if RADIAL:
                    ff=tl.arange(0,16)
                    rn_min=tl.load(RT+(child[:,None]*16+ff[None,:])*2)
                    rn_max=tl.load(RT+(child[:,None]*16+ff[None,:])*2+1)
                    mass=tl.full((2,16),0.,QP.dtype.element_ty)
                    for bin in range(4):
                        mu=tl.load(RF+(row*16+ff)*6+2+bin)
                        size=((bin+1)*nq+3)//4-(bin*nq+3)//4
                        mass+=size*tl.maximum(0.,tl.maximum(rn_min-mu[None,:],mu[None,:]-rn_max))
                    qmax=tl.max(tl.load(RF+(row*16+ff)*6+1),0)
                    guard=64.*1.1920928955078125e-7*nq*(tl.max(rn_max,1)+qmax)
                    bound=tl.maximum(bound,tl.max(mass,1)-guard-2e-4*(1+threshold))
                l0=tl.sum(tl.where(sides==0,bound,0.),0);l1=tl.sum(tl.where(sides==1,bound,0.),0)
                if FIRST:
                    near=tl.where(l0<=l1,node*2,node*2+1);far=tl.where(l0<=l1,node*2+1,node*2)
                    if tl.maximum(l0,l1)<=threshold:
                        sp+=1;stack=tl.where(slot==sp,far,stack);lbs=tl.where(slot==sp,tl.maximum(l0,l1),lbs)
                    if tl.minimum(l0,l1)<=threshold:
                        sp+=1;stack=tl.where(slot==sp,near,stack);lbs=tl.where(slot==sp,tl.minimum(l0,l1),lbs)
                else:
                    # The tail radius is fixed: prioritize only the short prefix.
                    if l1<=threshold:
                        sp+=1;stack=tl.where(slot==sp,node*2+1,stack);lbs=tl.where(slot==sp,0.,lbs)
                    if l0<=threshold:
                        sp+=1;stack=tl.where(slot==sp,node*2,stack);lbs=tl.where(slot==sp,0.,lbs)
            else:
                tl.store(LEAVES+row*G+count,node-G);count+=1
        active=sp>=0
        if FIRST:active=active&(count<LIMIT)
    if FIRST:
        tl.store(STACK+row*32+slot,stack);tl.store(LOWER+row*32+slot,lbs);tl.store(SP+row,sp)
    tl.store(COUNTS+row,count);tl.store(VISITS+row*2,nodes)
