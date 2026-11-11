import os

import torch
import pickle
import argparse
import numpy as np
import random

from utils.dataloader import TransformerDataLoader
from utils.cellspace import CellSpace
from trainer import Trainer
from utils.loss import Loss


def data_prepare():
    train_start, eval_start, test_start = 0, 2000, 3000
    root_path = os.getcwd()
    # 1. get cell_space traj_data and ground_truth
    if args.dataset == "porto":
        traj_data = pickle.load(open(root_path + "/dataset/porto/porto_1w.pkl", 'rb'))
        cell_space = CellSpace(args.cell_size, args.cell_size, -8.7005, 41.1001, -8.5192, 41.2086)
        if args.metric == "haus":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/porto/traj_simi_dict_hausdorff.pkl", 'rb')))
        elif args.metric == "dfret":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/porto/traj_simi_dict_dfret.pkl", 'rb')))
        elif args.metric == "sspd":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/porto/traj_simi_dict_sspd.pkl", 'rb')))
        elif args.metric == "dtw":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/porto/traj_simi_dict_dtw.pkl", 'rb')))

    elif args.dataset == "geolife":
        traj_data = pickle.load(open(root_path + "/dataset/geolife/geolife_1w.pkl", 'rb'))
        cell_space = CellSpace(args.cell_size, args.cell_size, 116.25, 39.8, 116.5, 40.1)
        if args.metric == "haus":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/geolife/traj_simi_dict_haus.pkl", 'rb')))
        elif args.metric == "dfret":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/geolife/traj_simi_dict_dfret.pkl", 'rb')))
        elif args.metric == "sspd":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/geolife/traj_simi_dict_sspd.pkl", 'rb')))
        elif args.metric == "dtw":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/geolife/traj_simi_dict_dtw.pkl", 'rb')))

    elif args.dataset == "tdriver":
        traj_data = pickle.load(open(root_path + "/dataset/tdriver/tdriver_1w.pkl", 'rb'))
        cell_space = CellSpace(args.cell_size, args.cell_size, 115.9, 39.6, 117, 40.7)
        if args.metric == "haus":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/tdriver/traj_simi_dict_haus.pkl", 'rb')))
        elif args.metric == "dfret":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/tdriver/traj_simi_dict_dfret.pkl", 'rb')))
        elif args.metric == "sspd":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/tdriver/traj_simi_dict_sspd.pkl", 'rb')))
        elif args.metric == "dtw":
            dis_matrix = torch.tensor(
                pickle.load(open(root_path + "/dataset/tdriver/traj_simi_dict_dtw.pkl", 'rb')))
    else:
        raise ValueError("错误数据集")

    dis_matrix = dis_matrix + dis_matrix.T
    train_data = (traj_data[train_start:eval_start], dis_matrix[train_start:eval_start, train_start:eval_start])
    eval_data = (traj_data[eval_start:test_start], dis_matrix[eval_start:test_start, eval_start:test_start])
    test_data = (traj_data[test_start:], dis_matrix[test_start:, test_start:])

    return cell_space, train_data, eval_data, test_data


def go():
    cellspace, train_data, eval_data, test_data = data_prepare()

    train_dataloader = TransformerDataLoader(train_data[0], train_data[1], mode="train",
                                             batch_size=args.bs, sampling_num=args.sampling_num,
                                             cellspace=cellspace, alpha=args.alpha)

    eval_dataloader = TransformerDataLoader(eval_data[0], eval_data[1], mode="eval", batch_size=20, sampling_num=20,
                                            cellspace=cellspace, alpha=args.alpha)

    test_dataloader = TransformerDataLoader(test_data[0], test_data[1], mode="test", batch_size=20, sampling_num=20,
                                            cellspace=cellspace, alpha=args.alpha)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    loss = Loss(args.loss)

    from Model.AggTransformer import AggAttnEncoder
    model = AggAttnEncoder(args.emb_dim, args.nhead, args.nlayer, args.dropout, args.mu, args.metric).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(args, model, optimizer, loss, device)

    trainer.run(train_dataloader, eval_dataloader, test_dataloader)


def seed_(seed):
    random.seed(seed)

    # Numpy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 使得每次生成的随机数可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--nlayer", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--cell_size", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)

    # training
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--train_mode", type=str, default="both", choices=["embedding", "pairwise", "both"])
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--bs", type=int, default=10)
    parser.add_argument("--sampling_num", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--dataset", type=str, default="porto", choices=["porto", "geolife", "tdriver"])
    parser.add_argument("--metric", type=str, default="haus", choices=["haus", "sspd", "dfret", "dtw"])
    parser.add_argument("--loss", type=str, default="euc", choices=["cheb", "euc"])

    parser.add_argument("--save_best", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=108)

    # model params
    args = parser.parse_args()
    print(args)
    seed_(args.seed)
    go()
