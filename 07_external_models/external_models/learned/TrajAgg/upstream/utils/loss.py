from torch.nn import Module
from torch import nn
import torch
import torch.nn.functional as F

class Loss(Module):
    def __init__(self, measure="euc"):
        super(Loss, self).__init__()
        self.measure = measure
        self.mse = nn.MSELoss()

    def embedding_learning_loss(self, traj_emb, truth_simi):
        if self.measure == "euc":
            # 计算相似度，并取exp
            pred_simi = torch.exp(-torch.cdist(traj_emb, traj_emb, 1))
            pred_simi = pred_simi[torch.triu(torch.ones(pred_simi.shape), diagonal=1) == 1]
            truth_simi = torch.tensor(truth_simi[torch.triu(torch.ones(truth_simi.shape), diagonal=1) == 1])
            train_loss = self.mse(pred_simi, truth_simi)

        elif self.measure == "cheb":
            pre_simi = torch.exp(-self.chebyshev_distance_matrix(traj_emb))
            pre_simi = pre_simi[torch.triu(torch.ones(pre_simi.shape), diagonal=1) == 1]
            truth_simi = torch.tensor(truth_simi[torch.triu(torch.ones(truth_simi.shape), diagonal=1) == 1])
            train_loss = self.mse(pre_simi, truth_simi)
        else:
            raise ValueError("measure must be euc or chebyshev")
        return train_loss

    def pairwise_learning_loss(self, traj_vecs1, traj_vecs2, target_dist):
        if self.measure == "euc":
            pred_simi = torch.norm((traj_vecs1 - traj_vecs2), p=1, dim=-1)
            pred_simi = torch.exp(-pred_simi)  # Map to a similarity space of 0-1
            loss = self.mse(pred_simi, target_dist)

        elif self.measure == "cheb":
            pred_simi = self.chebyshev_distance(traj_vecs1, traj_vecs2)
            pred_simi = torch.exp(-pred_simi)
            loss = self.mse(pred_simi, target_dist)

        else:
            raise ValueError("measure must be euc or chebyshev")

        return loss

    def train_loss(self, traj_vecs1, traj_vecs2, target_dist, sub_simi_matrix):
        if self.measure == "euc":
            # 计算pairwise learning 损失
            pred_simi = torch.norm((traj_vecs1 - traj_vecs2), p=1, dim=-1)
            pred_simi = torch.exp(-pred_simi)  # Map to a similarity space of 0-1
            loss1 = self.mse(pred_simi, target_dist)

            # 计算embedding-learning loss
            pred_simi1 = torch.exp(-torch.cdist(traj_vecs2, traj_vecs2, p=1))
            pred_simi1 = pred_simi1[torch.triu(torch.ones(pred_simi1.shape), diagonal=1) == 1]
            sub_simi_matrix = sub_simi_matrix[torch.triu(torch.ones(sub_simi_matrix.shape), diagonal=1) == 1]
            loss2 = self.mse(pred_simi1, sub_simi_matrix)
            loss = loss1 + loss2

        elif self.measure == "cheb":
            # 计算pairwise learning 损失
            pred_simi = self.chebyshev_distance(traj_vecs1, traj_vecs2)
            pred_simi = torch.exp(-pred_simi)
            loss1 = self.mse(pred_simi, target_dist)

            # 计算embedding-learning loss
            pred_simi1 = torch.exp(-self.chebyshev_distance_matrix(traj_vecs2))
            pred_simi1 = pred_simi1[torch.triu(torch.ones(pred_simi1.shape), diagonal=1) == 1]
            sub_simi_matrix = sub_simi_matrix[torch.triu(torch.ones(sub_simi_matrix.shape), diagonal=1) == 1]
            loss2 = self.mse(pred_simi1, sub_simi_matrix)
            loss = loss1 + loss2

        else:
            raise ValueError("measure must be euc or chebyshev")

        return loss



    def mseLoss(self, traj_vecs1, traj_vecs2, target_dist):
        predict_dis_matrix = torch.norm((traj_vecs1 - traj_vecs2), p=1, dim=-1)
        pred_dist = torch.exp(-predict_dis_matrix)  # Map to a similarity space of 0-1
        mse = self.mse(pred_dist, target_dist)
        return mse

    def chebyshev_distance(self, v1, v2):
        diffs = torch.abs(v1 - v2)
        max_diffs = torch.max(diffs, dim=-1).values
        return max_diffs

    def chebyshev_distance_matrix(self, X):
        # expand the dimension of X to shape (batch_size, 1, hidden_dim)
        X_expanded = X.unsqueeze(1)
        # expand the dimension of X to shape (1, batch_size, hidden_dim)
        X_tiled = X.unsqueeze(0)
        # calculate the absolute difference between X_expanded and X_tiled along the hidden_dim dimension
        diffs = torch.abs(X_expanded - X_tiled)
        # calculate the Chebyshev distance matrix on hidden_dim dimension
        distance_matrix = torch.max(diffs, dim=2).values
        return distance_matrix