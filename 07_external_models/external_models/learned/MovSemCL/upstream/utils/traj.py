import sys
sys.path.append('..')
import numpy as np
import random
import math

from config import Config
from utils import tool_funcs
from utils.rdp import rdp
from utils.cellspace import CellSpace
from utils.tool_funcs import truncated_rand


def straight(src):
    return src


def simplify(src):
    return rdp(src, epsilon = Config.traj_simp_dist)


def shift(src):
    return [[p[0] + truncated_rand(), p[1] + truncated_rand()] for p in src]


def mask(src):
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace = False)
    return np.delete(arr, mask_idx, 0).tolist()


def adaptive_mask(src):
    """Adaptive masking based on directional changes in trajectory"""
    if len(src) < 3:
        return mask(src) 
    l = len(src)
    arr = np.array(src)
    
    direction_changes = []
    for i in range(1, l - 1):
        v1 = arr[i] - arr[i-1]
        v2 = arr[i+1] - arr[i]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 0 and norm2 > 0:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)  # handle numerical errors
            angle_change = np.arccos(cos_angle)
        else:
            angle_change = 0
        
        direction_changes.append(angle_change)
    
    if len(direction_changes) > 0:
        max_change = max(direction_changes)
        if max_change > 0:
            direction_changes = [change / max_change for change in direction_changes]
        else:
            direction_changes = [0] * len(direction_changes)
    
    weights = []
    for i in range(l):
        if i == 0 or i == l - 1:
            weight = Config.adaptive_mask_endpoint_weight
        elif i - 1 < len(direction_changes):
            direction_score = direction_changes[i - 1]
            weight = Config.adaptive_mask_base_weight + (1 - direction_score) * Config.adaptive_mask_direction_factor
        else:
            weight = Config.adaptive_mask_base_weight
        weights.append(weight)
    
    total_weight = sum(weights)
    if total_weight > 0:
        probabilities = [w / total_weight for w in weights]
    else:
        probabilities = [1.0 / l] * l
    
    num_to_mask = int(l * Config.traj_mask_ratio)
    if num_to_mask >= l:
        num_to_mask = l - 1 
    mask_idx = np.random.choice(l, num_to_mask, replace=False, p=probabilities)
    return np.delete(arr, mask_idx, 0).tolist()


def subset(src):
    l = len(src)
    max_start_idx = l - int(l * Config.traj_subset_ratio)
    start_idx = random.randint(0, max_start_idx)
    end_idx = start_idx + int(l * Config.traj_subset_ratio)
    return src[start_idx: end_idx]


def get_aug_fn(name: str):
    return {'straight': straight, 'simplify': simplify, 'shift': shift,
            'mask': mask, 'adaptive_mask': adaptive_mask, 'subset': subset}.get(name, None)


def merc2cell2(src, cs: CellSpace):
    tgt = [ (cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p


def generate_spatial_features(src, cs: CellSpace):
    tgt = []
    
    if len(src) == 1:
        return [[0.0, 0.0, 0.0, 0.0]]
    
    if len(src) == 2:
        dx = (src[1][0] - src[0][0]) / (cs.x_max - cs.x_min)
        dy = (src[1][1] - src[0][1]) / (cs.y_max - cs.y_min)
        return [[0.0, 0.0, 0.0, 0.0], [dx, dy, 0.0, 0.0]]
    
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i-1] + lens[i]) / 2
        dist = dist / (Config.semmovcl_local_mask_sidelen / 1.414) # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i-1][0] - src[i][0],  src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0],  src[i+1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        dx = (src[i][0] - src[i-1][0]) / (cs.x_max - cs.x_min)
        dy = (src[i][1] - src[i-1][1]) / (cs.y_max - cs.y_min)
        tgt.append( [dx, dy, dist, radian] )

    tgt.insert(0, [0.0, 0.0, 0.0, 0.0] )
    
    dx = (src[-1][0] - src[-2][0]) / (cs.x_max - cs.x_min)
    dy = (src[-1][1] - src[-2][1]) / (cs.y_max - cs.y_min)
    tgt.append( [dx, dy, 0.0, 0.0] )
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length
