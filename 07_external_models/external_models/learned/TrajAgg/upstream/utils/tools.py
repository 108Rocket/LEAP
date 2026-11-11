import math
import torch
import numpy as np

def lonlat2meters(lon, lat):
    R = 6378137.0  # Earth's radius (WGS84)

    # Convert degrees to radians
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)

    # Web Mercator projection
    x = R * lon_rad
    y = R * math.log(math.tan(math.pi / 4 + lat_rad / 2))

    return x, y


def print_stats(trajs):
    lons = []
    lats = []
    for traj in trajs:
        for p in traj:
            lon, lat = p[0], p[1]
            lons.append(lon)
            lats.append(lat)
    lons = np.array(lons)
    lats = np.array(lats)
    mean_lon, mean_lat, std_lon, std_lat = np.mean(lons), np.mean(lats), np.std(lons), np.std(lats)
    x = {"mean_lon": mean_lon, "mean_lat": mean_lat, "std_lon": std_lon, "std_lat": std_lat}
    return x

# 从墨卡托投影得到轨迹xy坐标
def merc2cell2(src, cs):
    # convert and remove consecutive duplicates
    tgt = [(cs.get_xyidx_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i - 1][0]]
    tgt_xy, tgt_p = zip(*tgt)
    return torch.tensor(tgt_xy), torch.stack(tgt_p, dim=0)


def mean_pooling(x, padding_masks):
    """
    input: batch_size, seq_len, hidden_dim
    output: batch_size, hidden_dim
    """
    x = x * padding_masks.unsqueeze(-1)
    x = torch.sum(x, dim=1)/torch.sum(padding_masks, dim=1).unsqueeze(-1)  #  mean pooling excluding the padding part.
    return x


def creat_padding_mask(trajs):
    """Create a mask for a batch of trajectories.
    - False indicates that the position is a padding part that exceeds the original trajectory length
    - while True indicates that the position is the valid part of the trajectory.
    """
    lengths = torch.tensor([len(traj) for traj in trajs])
    max_len = max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
    return ~mask