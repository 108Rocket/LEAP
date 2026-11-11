import time
import torch
from tqdm import tqdm
from utils.top_k import top_k
import torch.nn.functional as F

class Trainer:
    def __init__(self, args, model, optimizer, loss, device):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.save_best = args.save_best
        self.target_measure = args.metric
        self.save_path = "/data/data_666/zhx111/TrajDIT_bridge/exp/snapshot/ddbm_best.pth"


    def run(self, train_dataloader, eval_dataloader, test_dataloader):

        best_score = 0
        for epoch in range(self.args.epoch):
            self.epoch = epoch
            self.train(train_dataloader)
            # evaluation
            eval_score = self.eval(eval_dataloader, stage="Eval")
            if eval_score[1] > best_score:
                best_score = eval_score[1]
                print(f"[Epoch {self.epoch}] Best Eval Score Updated, Test Model on Test Dataset...")
                test_score = self.eval(test_dataloader, stage="Test")
        return

    def train(self, train_dataloader):
        dataloader = train_dataloader.get_dataloader()
        start_time = time.time()
        self.model.train()
        epoch_loss = 0

        for batch in dataloader:
            if self.args.train_mode == "embedding":
                anchor, target, batch_distances, sub_simi = batch
                # move data to gpu
                target = tuple(item.to(self.device) for item in target)
                sub_simi = sub_simi.to(self.device)

                target_vec = self.model(target[0].float(), target[1].float(), None, target[2])
                loss = self.loss.embedding_learning_loss(target_vec, sub_simi)  # mse loss

            elif self.args.train_mode == "both":
                anchor, target, batch_distances, sub_simi = batch
                # move data to gpu
                anchor = tuple(item.to(self.device) for item in anchor)
                target = tuple(item.to(self.device) for item in target)
                batch_distances = batch_distances.to(self.device)
                sub_simi = sub_simi.to(self.device)

                anchor_vec = self.model(anchor[0].float(), anchor[1].float(), None, anchor[2])
                target_vec = self.model(target[0].float(), target[1].float(), None, target[2])
                loss = self.loss.train_loss(anchor_vec.float(), target_vec.float(), batch_distances.float(), sub_simi)  # mse loss

            elif self.args.train_mode == "pairwise":
                anchor, target, batch_distances, sub_simi = batch
                # move data to gpu
                anchor = tuple(item.to(self.device) for item in anchor)
                target = tuple(item.to(self.device) for item in target)
                batch_distances = batch_distances.to(self.device)

                anchor_vec = self.model(anchor[0].float(), anchor[1].float(), None, anchor[2])
                target_vec = self.model(target[0].float(), target[1].float(), None, target[2])
                loss = self.loss.pairwise_learning_loss(anchor_vec.float(), target_vec.float(), batch_distances.float())

            else:
                raise ValueError(f"Train mode {self.args.train_mode} not supported")

            epoch_loss += loss.item()  # total loss of this epoch

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        end_time = time.time()
        print(f"[Epoch {self.epoch}] Epoch Loss:{epoch_loss}, Train Time:{end_time - start_time}")

    def eval(self, eval_dataloader, stage="Eval"):
        dataloader = eval_dataloader.get_dataloader()
        dis_matrix = eval_dataloader.get_dis_matrix()
        # 1. Obtain trajectory representation vector
        self.model.eval()
        emds = torch.zeros(len(dataloader.dataset), self.model.d_model).to(self.device)  # batch_size * hidden_dim
        with torch.no_grad():
            for batch in dataloader:
                moved_batch = tuple(item.to(self.device) for item in batch)
                trajs_p, trajs_xy, padding_masks, IDs = moved_batch
                # _time = time.time()
                traj_vecs = self.model(trajs_p.float(), trajs_xy.float(), None, padding_masks)
                # print(time.time() - _time)
                emds[IDs] = traj_vecs

        # 2. calculate top-k acc
        topk_acc = top_k(emds.cpu(), dis_matrix, self.loss.measure)

        if stage == "Eval":
            print(f"[Epoch {self.epoch}] {stage} {self.target_measure.upper()}@Top-k Acc:{topk_acc}")

        if stage == "Test":
            if self.save_best:
                self.traj_embs = emds.cpu()
                self.checkpoint_epoch = self.epoch
            print(f"[Epoch {self.epoch}] |-> {stage} {self.target_measure.upper()}@Top-k Acc:{topk_acc}")
        return topk_acc

    def save_model(self):
        state = {'model_state': self.model.state_dict()}
        torch.save(state, self.save_path)
        return

    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.model.load_state_dict(checkpoint['model_state'])
        return

