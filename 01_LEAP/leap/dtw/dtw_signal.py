"""Ordered radial signals: min_path sum max_probe |r_j(p)-r_j(q)| <= DTW.

No point is dropped or reordered. Full-length alignment is not a constant-size
trajectory encoding. Hard-path differentiation uses a selected minimizing path
and maximizing probe, giving the branch gradient away from ties. See
docs/dtw-ordered-probes.md, (S1)-(S3).
"""
import torch
import triton
import triton.language as tl

_ENCODE_CHUNK_POINTS = 262144
_TRACE_MAX_BYTES = 4 * 1024**3


def encode(index, theta):
    points = index.points
    if (points.is_cuda and theta.device == points.device and theta.dtype == torch.float32
            and theta.ndim == 2 and theta.shape[1] == 2 and theta.is_contiguous()
            and not (torch.is_grad_enabled() and (points.requires_grad or theta.requires_grad))):
        from leap.common import _query_encoding
        with torch.cuda.device(points.device):
            if _query_encoding.supported(index):
                return _query_encoding.signals(index, theta)
    def values(part):
        return (part[:, None] - theta[None]).square().sum(-1).sqrt()
    if len(points) <= _ENCODE_CHUNK_POINTS:
        return values(points).contiguous()
    if torch.is_grad_enabled() and (points.requires_grad or theta.requires_grad):
        return torch.cat([values(part) for part in points.split(_ENCODE_CHUNK_POINTS)]).contiguous()
    # The final signal is required; the all-points broadcast intermediates are not.
    output = torch.empty((len(points), len(theta)), device=points.device,
                         dtype=torch.promote_types(points.dtype, theta.dtype))
    for lo in range(0, len(points), _ENCODE_CHUNK_POINTS):
        hi = min(lo + _ENCODE_CHUNK_POINTS, len(points))
        output[lo:hi] = values(points[lo:hi])
    return output


@triton.jit
def signal_kernel(A, AO, B, BO, Q, D, OUT, WORK, TRACE,
                  M: tl.constexpr, BLOCK: tl.constexpr, TRACEBACK: tl.constexpr,
                  TRACE_ROWS: tl.constexpr = 0):
    pair = tl.program_id(0)
    height: tl.constexpr = TRACE_ROWS if TRACE_ROWS else BLOCK
    q, d = tl.load(Q + pair), tl.load(D + pair)
    a0, b0 = tl.load(AO + q), tl.load(BO + d)
    na, nb = tl.load(AO + q + 1) - a0, tl.load(BO + d + 1) - b0
    j, f = tl.arange(0, BLOCK), tl.arange(0, M)
    br = tl.load(B + (b0+j[:, None])*M+f[None], j[:, None]<nb, other=0.)
    work = WORK + pair*3*BLOCK
    for bank in range(3):
        tl.store(work+bank*BLOCK+j, float('inf'))
    tl.debug_barrier()
    for diagonal in range(na+nb-1):
        i = diagonal-j
        valid = (i>=0)&(i<na)&(j<nb)
        ar = tl.load(A+(a0+i[:, None])*M+f[None], valid[:, None], other=0.)
        diff = tl.abs(ar-br)
        cost = tl.max(diff, 1)
        p1, p2 = (diagonal+2)%3, (diagonal+1)%3
        up = tl.load(work+p1*BLOCK+j, volatile=True)
        left = tl.load(work+p1*BLOCK+j-1, j>0, other=float('inf'), volatile=True)
        cross = tl.load(work+p2*BLOCK+j-1, j>0, other=float('inf'), volatile=True)
        pred = tl.minimum(cross, tl.minimum(up, left))
        pred = tl.where((i==0)&(j==0), 0., pred)
        value = tl.where(valid, cost+pred, float('inf'))
        if TRACEBACK:
            # Prefer diagonal, then up, then left when predecessor costs tie.
            direction = tl.where((cross<=up)&(cross<=left), 0, tl.where(up<=left, 1, 2))
            winner = tl.min(tl.where(diff==cost[:, None], f[None], M), 1)
            tl.store(TRACE+(pair*height+i)*BLOCK+j, winner*4+direction, valid)
        tl.store(work+(diagonal%3)*BLOCK+j, value)
        tl.debug_barrier()
    value = tl.load(work+((na+nb-2)%3)*BLOCK+nb-1, volatile=True)
    tl.store(OUT+pair, value)


@triton.jit
def traceback_kernel(AO, BO, Q, D, TRACE, PATH_A, PATH_B, PROBE, COUNT,
                     BLOCK: tl.constexpr, WIDTH: tl.constexpr, TRACE_ROWS: tl.constexpr = 0):
    pair = tl.program_id(0)
    height: tl.constexpr = TRACE_ROWS if TRACE_ROWS else BLOCK
    q, d = tl.load(Q+pair), tl.load(D+pair)
    a0, b0 = tl.load(AO+q), tl.load(BO+d)
    na, nb = tl.load(AO+q+1)-a0, tl.load(BO+d+1)-b0
    t = tl.arange(0, WIDTH)
    tl.store(PATH_A+pair*WIDTH+t, 0)
    tl.store(PATH_B+pair*WIDTH+t, 0)
    tl.store(PROBE+pair*WIDTH+t, 0)
    tl.debug_barrier()
    i, j, count = na-1, nb-1, 0
    running = True
    while running:
        code = tl.load(TRACE+(pair*height+i)*BLOCK+j).to(tl.int32)
        tl.store(PATH_A+pair*WIDTH+count, a0+i)
        tl.store(PATH_B+pair*WIDTH+count, b0+j)
        tl.store(PROBE+pair*WIDTH+count, code//4)
        count += 1
        running = (i!=0)|(j!=0)
        direction = code%4
        i -= tl.cast(direction!=2, tl.int32)
        j -= tl.cast(direction!=1, tl.int32)
    tl.store(COUNT+pair, count)


@torch.no_grad()
def pairs(q, db, qids, dids, theta=None, fields=None, chunk=4096):
    if fields is None:
        a = encode(q, theta)
        fields = (a, a if q is db else encode(db, theta))
    a, b = fields
    assert a.shape[1]==b.shape[1] and triton.next_power_of_2(a.shape[1])==a.shape[1]
    assert qids.shape==dids.shape and qids.ndim==1
    # Use the existing inference buckets for training mining/validation as well.
    from leap.dtw.dtw_signal_bucket import pairs as bucketed_pairs
    return bucketed_pairs(q, db, qids, dids, fields=fields, chunk=chunk)


def differentiable_pairs(q, db, qids, dids, theta, fields=None):
    with torch.no_grad():
        if fields is None:
            a = encode(q, theta)
            fields = (a, a if q is db else encode(db, theta))
        a, b = fields
        na = q.offsets[qids+1]-q.offsets[qids]
        nb = db.offsets[dids+1]-db.offsets[dids]
        width = 2*triton.next_power_of_2(int(torch.maximum(na.max(), nb.max())))
        maximum = triton.next_power_of_2(int(nb.max()))
        count_pairs = len(qids)
        output = torch.empty(count_pairs, device=theta.device)
        path_a = torch.empty((count_pairs, width), device=theta.device, dtype=torch.int64)
        path_b = torch.empty_like(path_a)
        probe = torch.empty_like(path_a)
        counts = torch.empty(count_pairs, device=theta.device, dtype=torch.int32)
        lower = 0
        for block in (2**i for i in range(maximum.bit_length())):
            rows = ((nb > lower) & (nb <= block)).nonzero().flatten()
            lower = block
            if not len(rows):
                continue
            height = int(na[rows].max())
            if height*block > _TRACE_MAX_BYTES:
                raise MemoryError('A single DTW traceback exceeds the temporary memory budget')
            chunk = min(len(rows), _TRACE_MAX_BYTES // (height*block))
            trace = torch.empty((chunk, height, block), device=theta.device, dtype=torch.uint8)
            work = torch.empty((chunk, 3, block), device=theta.device)
            values = torch.empty(chunk, device=theta.device)
            pa = torch.empty((chunk, width), device=theta.device, dtype=torch.int64)
            pb, pr = torch.empty_like(pa), torch.empty_like(pa)
            ct = torch.empty(chunk, device=theta.device, dtype=torch.int32)
            for lo in range(0, len(rows), chunk):
                selected = rows[lo:lo+chunk]
                size = len(selected)
                qi, di = qids[selected].contiguous(), dids[selected].contiguous()
                signal_kernel[(size,)](a, q.offsets, b, db.offsets, qi, di, values, work, trace,
                    a.shape[1], block, True, height, num_warps=4, enable_fp_fusion=False)
                traceback_kernel[(size,)](q.offsets, db.offsets, qi, di, trace,
                    pa, pb, pr, ct, block, width, height, num_warps=4)
                output[selected] = values[:size]
                path_a[selected], path_b[selected] = pa[:size], pb[:size]
                probe[selected], counts[selected] = pr[:size], ct[:size]
            del trace, work, values, pa, pb, pr, ct
        # Restore original pair order and reduction width before the single backward pass.
        mask = torch.arange(width, device=theta.device)[None] < counts[:, None]
    c = theta[probe]
    ra = torch.linalg.vector_norm(q.points[path_a]-c, dim=-1)
    rb = torch.linalg.vector_norm(db.points[path_b]-c, dim=-1)
    score = ((ra-rb).abs()*mask).sum(-1)
    return score, output
