import numpy as np
import torch

from tools import feature_distance
from tools import function

def intersect_sizes(true_list, test_list):
    return np.array([len(np.intersect1d(true_value, list(test_value))) for true_value, test_value in zip(true_list, test_list)])


def test_all_print_new(query_id_list, base_id_list, true_knn, test_distance_matrix):
    
    test_knn = np.argsort(test_distance_matrix)

    top_test_dict = {1:[1], 5:[5], 10:[10], 50:[50, 10], 100:[100]}

    for tem_test_num in top_test_dict.keys():
        top_true_list  = top_test_dict[tem_test_num]
        test_top_id    = test_knn[:, :tem_test_num]
        intersect_list = [intersect_sizes(true_knn[:, :tem_true_num], test_top_id) / float(tem_true_num) for tem_true_num in top_true_list]
        recall_list    = [np.mean(tem_list) for tem_list in intersect_list]
        for pos in range(len(top_true_list)):
            if tem_test_num == 1 and top_true_list[pos] == 1:
                top1_recall = recall_list[pos] * 100
            if tem_test_num == 5 and top_true_list[pos] == 5:
                top5_recall = recall_list[pos] * 100
            if tem_test_num == 10 and top_true_list[pos] == 10:
                top10_recall = recall_list[pos] * 100
            if tem_test_num == 50 and top_true_list[pos] == 50:
                top50_recall = recall_list[pos] * 100
            if tem_test_num == 100 and top_true_list[pos] == 100:
                top100_recall = recall_list[pos] * 100
            if tem_test_num == 50 and top_true_list[pos] == 10:
                top10_50_recall = recall_list[pos] * 100
            
    total_list = []
    recall_list = [top1_recall, top5_recall, top10_recall, top50_recall, top100_recall, top10_50_recall]
    total_list.extend(recall_list)
    return total_list


def pivot_lb_scores_from_phi(
    query_phi: np.ndarray,   # (Q, M)
    base_phi: np.ndarray,    # (B, M)
    batch_size: int = 64
) -> np.ndarray:
    """
    Compute Pivot Lower Bound matrix:
        LB(q,c) = max_i |phi(q,i) - phi(c,i)|
    Returns:
        lb: (Q, B) float32
    """
    Q, M = query_phi.shape
    B = base_phi.shape[0]
    lb = np.empty((Q, B), dtype=np.float32)

    for qs in range(0, Q, batch_size):
        qe = min(qs + batch_size, Q)
        # (b,1,M) - (1,B,M) -> (b,B,M)
        diff = np.abs(query_phi[qs:qe, None, :] - base_phi[None, :, :])
        lb[qs:qe] = diff.max(axis=2).astype(np.float32)
    return lb

def pivot_lb_scores_from_phi_gpu(
    query_phi: np.ndarray,   # (Q, M)
    base_phi: np.ndarray,    # (B, M)
    base_batch_size: int = 10000, # 每次处理 1万个 Base，显存占用极小
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Optimized GPU Pivot Lower Bound computation for massive Base sets.
    
    Logic: LB(q,c) = L_infinity distance between q and c.
    Strategy: Keep Query on GPU, iterate over Base in chunks.
    """
    dev = function.resolve_torch_device(device)

    Q, M = query_phi.shape
    B = base_phi.shape[0]

    # 2. 将 Query 全部放入 GPU
    # Q=1000, M=128 -> 仅占用 ~0.5MB 显存，非常安全
    query_t = torch.as_tensor(query_phi, dtype=torch.float32, device=dev)

    # 3. 在 CPU 预分配结果矩阵 (1000 * 1000000 * 4 bytes ≈ 3.7 GB 内存)
    # 如果系统内存不够，需要换成 np.memmap 或分块写入磁盘
    lb = np.empty((Q, B), dtype=np.float32)

    # 4. 对 Base 进行分块循环 (防止显存爆炸的核心)
    # 我们不分切 Query，而是分切 Base，因为 Base 是一百万级
    for bs in range(0, B, base_batch_size):
        be = min(bs + base_batch_size, B)
        
        # 将当前批次的 Base 移入 GPU
        # (10000, 128) -> ~5MB 显存
        base_batch_t = torch.as_tensor(base_phi[bs:be], dtype=torch.float32, device=dev)

        # 5. 使用 torch.cdist 计算 L_infinity 距离
        # 这是 PyTorch 底层优化的算子，比手动广播减法快且省显存
        # p=float('inf') 对应的就是 max(|x - y|)
        # 结果形状: (Q, current_base_batch_size) -> (1000, 10000)
        dists = torch.cdist(query_t, base_batch_t, p=float('inf'))
        # 6. 转回 CPU 并填入结果
        lb[:, bs:be] = dists.cpu().numpy()
        
        # 手动清理显存（可选，防止碎片化）
        del base_batch_t, dists

    return lb

def get_feature_distance(query_embeddings, base_embeddings):
    return feature_distance.l2_dist(query_embeddings, base_embeddings)
