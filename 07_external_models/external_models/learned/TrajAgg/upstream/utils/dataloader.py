import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils.tools import print_stats, merc2cell2


class TransformerDataset(Dataset):
    def __init__(self, trajs):
        super(TransformerDataset, self).__init__()
        self.trajs = trajs
        self.wgs = trajs.wgs_seq.tolist()
        self.merc = trajs.merc_seq.tolist()
        self.IDs = [i for i in range(len(self.trajs))]

    def __getitem__(self, idx):
        return torch.tensor(self.wgs[idx]), torch.tensor(self.merc[idx]), idx

    def get_items(self, indices):
        wgs = [self.__getitem__(i)[0] for i in indices]
        merc = [self.__getitem__(i)[1] for i in indices]
        return wgs, merc

    def __len__(self):
        return len(self.IDs)


class TransformerDataLoader():
    def __init__(self, trajs, dis_matrix=None, batch_size=20, mode="train", sampling_num=20, alpha=16, cellspace=None):
        self.dataset = TransformerDataset(trajs)
        self.data_range = print_stats(self.dataset.merc)
        self.mean_lon = self.data_range["mean_lon"]
        self.mean_lat = self.data_range["mean_lat"]
        self.std_lon = self.data_range["std_lon"]
        self.std_lat = self.data_range["std_lat"]

        self.mode = mode
        self.dis_matrix = dis_matrix
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.alpha = alpha
        self.dataloader = self.create_dataloader()
        self.cellspace = cellspace

    def get_dataloader(self):
        return self.dataloader

    def get_dis_matrix(self):
        return self.dis_matrix

    def collate_fn_train(self, data):
        *_, indices = zip(*data)

        anchor_idxs = []
        target_idxs = []
        batch_distances = []
        for anchor_idx in indices:
            target_indices, sims = self.random_sampling(anchor_idx)
            batch_distances.extend(sims)
            anchor_idxs.extend([anchor_idx] * self.sampling_num)
            target_idxs.extend(target_indices)
        batch_distances = torch.tensor(batch_distances)

        # anchor
        _, anchor_trajs_merc = self.dataset.get_items(anchor_idxs)
        anchor_trajs_xy, anchor_trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in anchor_trajs_merc])
        anchor_trajs_p = self.trajs_normalize(anchor_trajs_p)
        anchor_trajs_mask = self.creat_padding_mask(anchor_trajs_p)
        anchor_trajs_p = pad_sequence(anchor_trajs_p, batch_first=True, padding_value=0.0)
        anchor_trajs_xy = pad_sequence(anchor_trajs_xy, batch_first=True, padding_value=0.0)

        # target
        _, target_trajs_merc = self.dataset.get_items(target_idxs)
        target_trajs_xy, target_trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in target_trajs_merc])
        target_trajs_p = self.trajs_normalize(target_trajs_p)
        target_trajs_mask = self.creat_padding_mask(target_trajs_p)
        target_trajs_p = pad_sequence(target_trajs_p, True, padding_value=0.0)
        target_trajs_xy = pad_sequence(target_trajs_xy, True, padding_value=0.0)

        anchor = (anchor_trajs_p, anchor_trajs_xy, anchor_trajs_mask)
        target = (target_trajs_p, target_trajs_xy, target_trajs_mask)

        sub_simi = self.dis_matrix[target_idxs][:, target_idxs]
        sub_simi = torch.exp(- self.alpha * sub_simi)

        return anchor, target, batch_distances, sub_simi


    # test and eval dataloader
    def collate_fn_test_eval(self, data):
        *_, indices = zip(*data)
        _, trajs_merc = self.dataset.get_items(indices)
        trajs_xy, trajs_p = zip(*[merc2cell2(t, self.cellspace) for t in trajs_merc])
        trajs_p = self.trajs_normalize(trajs_p)
        trajs_mask = self.creat_padding_mask(trajs_p)
        trajs_p = pad_sequence(trajs_p, batch_first=True)
        trajs_xy = pad_sequence(trajs_xy, batch_first=True)

        return trajs_p, trajs_xy, trajs_mask, torch.tensor(indices)


    def trajs_normalize(self, trajs):
        mean = torch.tensor([self.mean_lon, self.mean_lat], dtype=torch.float32)
        std = torch.tensor([self.std_lon, self.std_lat], dtype=torch.float32)
        normalized_trajs = [(traj - mean) / std for traj in trajs]
        return normalized_trajs

    def random_sampling(self, idx):
        N = len(self.dis_matrix)
        distances = self.dis_matrix[idx]
        similarity = torch.exp(-self.alpha * distances)  # 1*N
        indices = np.random.choice(range(N), size=self.sampling_num, replace=False)
        return indices, similarity[indices]

    # def random_sampling(self, idx):
    #     N = len(self.dis_matrix)
    #     distances = self.dis_matrix[idx]  # 1 x N
    #     similarity = torch.exp(-self.alpha * distances)  # 1 x N
    #
    #     # 排除自己
    #     candidate_indices = [i for i in range(N) if i != idx]
    #
    #     # 计算采样数量的一半
    #     half_num = self.sampling_num // 2
    #
    #     # 将候选索引按距离排序
    #     sorted_indices = sorted(candidate_indices, key=lambda i: distances[i])
    #
    #     # 前 half_num 是最相似的（距离小）
    #     most_similar = sorted_indices[:half_num]
    #
    #     # 后 half_num 是最不相似的（距离大）
    #     least_similar = sorted_indices[-half_num:]
    #
    #     # 合并两个部分
    #     selected_indices = most_similar + least_similar
    #
    #     # 返回索引和对应的相似度
    #     return selected_indices, similarity[selected_indices]



    @staticmethod
    def creat_padding_mask(trajs):
        """Create a mask for a batch of trajectories.
        - False indicates that the position is a padding part that exceeds the original trajectory length
        - while True indicates that the position is the valid part of the trajectory.
        """
        lengths = torch.tensor([len(traj) for traj in trajs])
        max_len = max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        return ~mask


    def create_dataloader(self) -> DataLoader:
        """
        given trajectory dataset and batch_size, return the corresponding DataLoader
        """
        pairs_num = self.batch_size * self.sampling_num  # calculate all pairs required for training

        if self.mode == "train":
            dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=32,
                                    pin_memory=True, collate_fn=self.collate_fn_train)
            print(
                f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples, {pairs_num} samples per batch")
            return dataloader

        if self.mode == "test" or self.mode == "eval":  # do not need to construct pairs, just encoding
            # To keep the batch_size consistent with training, set the batch_size to be batch_size * sampling_num.
            dataloader = DataLoader(dataset=self.dataset, batch_size=pairs_num, shuffle=False, num_workers=32,
                                    pin_memory=True, collate_fn=self.collate_fn_test_eval)
            print(
                f"[Data Preparation] TransformerDataLoader: batch size {self.batch_size}, {len(dataloader.dataset)} samples, {pairs_num} samples per batch")
            return dataloader

