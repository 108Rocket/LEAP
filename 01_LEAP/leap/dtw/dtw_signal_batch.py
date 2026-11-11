"""Equivalent batched signal DTW with immutable length metadata."""
import torch
import triton

from leap.dtw.dtw_signal import encode, signal_kernel


class SignalPairPlan:
    def __init__(self, database):
        self.offsets = database.offsets
        self.version = self.offsets._version
        lengths = self.offsets[1:] - self.offsets[:-1]
        maximum = triton.next_power_of_2(int(lengths.max()))
        self.blocks = [2**i for i in range(maximum.bit_length())]
        boundaries = torch.tensor(self.blocks, device=self.offsets.device)
        self.buckets = torch.bucketize(lengths, boundaries)

    @torch.no_grad()
    def pairs(self, q, db, qids, dids, theta=None, fields=None, chunk=16384):
        from leap.common.runtime.context import current
        runtime_plan = current()
        if runtime_plan is not None and fields is not None:
            return runtime_plan.signal_values(q, db, qids, dids, fields)
        if db.offsets is not self.offsets or self.offsets._version != self.version:
            raise ValueError('Database offsets changed; rebuild SignalPairPlan')
        a, b = (encode(q, theta), encode(db, theta)) if fields is None else fields
        out = torch.empty(dids.shape, device=dids.device, dtype=torch.float32)
        if not len(dids):
            return out
        bins = self.buckets[dids]
        # Collect all sizes in one synchronization; scores are never cached.
        counts = torch.bincount(bins, minlength=len(self.blocks)).cpu().tolist()
        order = torch.argsort(bins)
        start = 0
        for block, count in zip(self.blocks, counts):
            selected = order[start:start+count]
            start += count
            if not count:
                continue
            qs, ds = qids[selected].contiguous(), dids[selected].contiguous()
            values = torch.empty(count, device=dids.device, dtype=torch.float32)
            work = torch.empty((min(chunk, count), 3, block), device=dids.device)
            warps = 4
            if a.shape[1] == 16:
                warps = 1 if block <= 64 else (2 if block == 128 else 4)
            for lo in range(0, count, chunk):
                hi = min(lo+chunk, count)
                signal_kernel[(hi-lo,)](
                    a, q.offsets, b, db.offsets, qs[lo:hi], ds[lo:hi],
                    values[lo:hi], work, work, a.shape[1], block, False,
                    num_warps=warps, enable_fp_fusion=False)
            out[selected] = values
        return out
