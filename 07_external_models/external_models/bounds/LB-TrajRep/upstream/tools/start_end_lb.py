# tools/start_end_lb.py
import numpy as np
import torch

from tools import function

def start_end_lb(query_trajs, base_trajs):
    """
    query_trajs, base_trajs: list of traj, each traj shape (L,2)
    return:
        lb: (Q,B) matrix
    """
    Q = len(query_trajs)
    B = len(base_trajs)

    qs = np.array([t[0] for t in query_trajs])   # (Q,2)
    qe = np.array([t[-1] for t in query_trajs])  # (Q,2)
    bs = np.array([t[0] for t in base_trajs])    # (B,2)
    be = np.array([t[-1] for t in base_trajs])   # (B,2)

    # broadcast
    ds = qs[:, None, :] - bs[None, :, :]
    de = qe[:, None, :] - be[None, :, :]

    lb = np.maximum(
        np.sqrt((ds * ds).sum(axis=2)),
        np.sqrt((de * de).sum(axis=2)),
    )
    return lb

def start_end_lb_gpu(
    query_trajs: list, 
    base_trajs: list, 
    device: str = "cuda:0",
    batch_size: int = 500000 
) -> np.ndarray:
    """
    【高精度版】GPU 加速 SE-LB 计算。
    
    此版本完美复刻原始 NumPy 代码的逻辑，确保结果完全一致：
    1. 使用 Float64 (双精度) 避免精度丢失。
    2. 使用手动广播减法 `(A-B)^2` 代替 `torch.cdist`。
    """
    dev = function.resolve_torch_device(device)

    Q = len(query_trajs)
    B = len(base_trajs)

    # 1. 预处理 (CPU): 提取坐标，保持原始 float64 精度
    # 注意：这里不再使用 .astype(np.float32)
    qs_np = np.stack([t[0] for t in query_trajs])
    qe_np = np.stack([t[-1] for t in query_trajs])
    bs_np = np.stack([t[0] for t in base_trajs])
    be_np = np.stack([t[-1] for t in base_trajs])

    # 2. 将坐标数据以 Float64 格式移入 GPU
    qs = torch.from_numpy(qs_np).to(dev, dtype=torch.float64)
    qe = torch.from_numpy(qe_np).to(dev, dtype=torch.float64)
    bs = torch.from_numpy(bs_np).to(dev, dtype=torch.float64)
    be = torch.from_numpy(be_np).to(dev, dtype=torch.float64)

    # 3. 预分配 CPU 结果矩阵
    # 最终结果通常是 float32，所以这里可以分配 float32
    lb = np.empty((Q, B), dtype=np.float32)

    # 4. 分块计算
    for i in range(0, B, batch_size):
        end = min(i + batch_size, B)
        
        # 切片当前批次的 base 点
        bs_chunk = bs[i:end]
        be_chunk = be[i:end]

        # === 核心计算：手动广播，精确复刻 NumPy 逻辑 ===
        # qs: (Q, 2) -> (Q, 1, 2)
        # bs_chunk: (batch, 2) -> (1, batch, 2)
        # ds_chunk: (Q, batch, 2)
        ds_chunk = qs.unsqueeze(1) - bs_chunk.unsqueeze(0)
        de_chunk = qe.unsqueeze(1) - be_chunk.unsqueeze(0)

        # 计算平方和 (欧氏距离的平方)
        # 结果形状: (Q, batch)
        d_start_sq = (ds_chunk ** 2).sum(dim=2)
        d_end_sq = (de_chunk ** 2).sum(dim=2)
        
        # 在 GPU 上开根号
        d_start = torch.sqrt(d_start_sq)
        d_end = torch.sqrt(d_end_sq)

        # 取最大值
        res = torch.maximum(d_start, d_end)

        # 将结果从 GPU 移回 CPU，并转换为 float32 存入
        lb[:, i:end] = res.cpu().numpy().astype(np.float32)

    return lb
