import triton
import triton.language as tl


@triton.jit
def merge_top20(PREV,IDS,VALUES,COUNTS,OFF,OUT,RADIUS,
                CAP:tl.constexpr,BLOCK:tl.constexpr,FIXED:tl.constexpr):
    tl.static_assert(FIXED)
    row=tl.program_id(0);i=tl.arange(0,32)
    count=tl.load(COUNTS+row)
    best=tl.load(PREV+row*20+i,i<20,other=9187343244130779135)
    for start in range(0,count,32):
        good=start+i<count
        d=tl.load(IDS+row*CAP+start+i,good,other=0)
        v=tl.load(VALUES+row*CAP+start+i,good,other=0.)
        key=(v.to(tl.int32,bitcast=True).to(tl.int64)<<32)|d.to(tl.int64)
        key=tl.where(good,key,9187343244130779135)
        best=tl.sort(tl.minimum(best,tl.sort(key,descending=True)),descending=False)
    tl.store(OUT+row*20+i,best,i<20)
    kth=tl.sum(tl.where(i==19,best,0),0)
    tl.store(RADIUS+row,(kth>>32).to(tl.int32).to(tl.float32,bitcast=True))
