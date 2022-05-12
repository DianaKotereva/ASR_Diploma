from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import gc
import numpy as np

class Trainer:
    def __init__(self, model,
                 optimizer,
                 train_dataloader,
                 val_dataloader,
                 tboard_log_dir: str = './tboard_logs/',
                 lr_scheduler=None,
                 lr_scheduler_type=None,
                 device = 'cpu'
                 ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.device = device
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.model = self.model.to(self.device)

        self.global_step = 0
        self.train_writer = SummaryWriter(log_dir=tboard_log_dir + "train/")
        self.val_writer = SummaryWriter(log_dir=tboard_log_dir + "val/")
        self.cache = self.cache_states()

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_type = lr_scheduler_type
        if lr_scheduler_type not in [None, 'per_batch', 'per_epoch']:
            raise ValueError("lr_scheduler_type must be one of: None, 'per_batch', 'per_epoch'. "
                             f"Not: {lr_scheduler_type}")
        self.model_name = tboard_log_dir.split('/')[1]
        
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def post_val_stage(self, val_loss):
        if self.lr_scheduler is not None and self.lr_scheduler_type == 'per_epoch':
            self.lr_scheduler.step(val_loss)

    def post_train_batch(self):
        if self.lr_scheduler is not None and self.lr_scheduler_type == 'per_batch':
            self.lr_scheduler.step()

    def train(self, num_epochs: int):
        train_loader = self.train_dataloader
        val_loader = self.val_dataloader
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.model.train()
            for batch in tqdm(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, details = self.model.compute_all(batch['batch'], num_epoch = epoch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.post_train_batch()
                gc.collect()
                torch.cuda.empty_cache()
                for k, v in details.items():
                    self.train_writer.add_scalar(
                        k, v, global_step=self.global_step)
                self.global_step += 1

            self.model.eval()
            print('Val')
            
            val_losses = []
#             print(len(val_losses))
            val_logs = defaultdict(list)
            for batch in tqdm(val_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_val, details = self.model.compute_all(batch['batch'], num_epoch = epoch)
                val_losses.append(loss_val.item())
                for k, v in details.items():
                    val_logs[k].append(v)

            val_logs = {k: np.mean(v) for k, v in val_logs.items()}
#             print('Means', val_logs)
            for k, v in val_logs.items():
                self.val_writer.add_scalar(k, v, global_step=self.global_step)

            val_acc = val_logs['segment_acc_score']
            val_loss = np.mean(val_losses)
#             print('Mean Loss', val_loss, np.min(val_losses), np.median(val_losses), np.max(val_losses))
#             print(val_losses[:10])

            self.post_val_stage(val_loss)

            if val_loss < best_loss:
                self.save_checkpoint(
                    "./"+self.model_name+"_best_checkpoint.pth")
                best_loss = val_loss
                best_acc = val_acc
        return best_acc

#     def calculate_val_score(self, val_loader):

#         model = self.model
#         model.eval()
#         ys_all = []
#         preds_all = []

#         val_logs = defaultdict(list)
#         for batch in tqdm(val_loader):
#             batch = {k: v.to(self.device) for k, v in batch.items()}
#             y, preds = model.eval_all(batch['batch'])
#             ys_all.append(y)
#             preds_all.append(preds)

#         ys_all = np.hstack(ys_all)
#         preds_all = np.vstack(preds_all)
        
#         return acc5, acc1

    def find_lr(self, min_lr: float = 1e-6,
                max_lr: float = 1e-1,
                num_lrs: int = 20,
                smooth_beta: float = 0.8) -> dict:
        # exp((1-t) * log(left) + t * log(right))
        lrs = np.geomspace(start=min_lr, stop=max_lr, num=num_lrs)
        logs = {'lr': [], 'loss': [], 'avg_loss': []}
        avg_loss = None
        train_loader = self.train_dataloader

        self.model.train()
        for lr, batch in tqdm(zip(lrs, train_loader), desc='finding LR', total=num_lrs):
            # apply new lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # train step
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, details = self.model.compute_all(batch['batch'], num_epoch = epoch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate smoothed loss
            if avg_loss is None:
                avg_loss = loss
            else:
                avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss

            # store values into logs
            logs['lr'].append(lr)
            logs['avg_loss'].append(avg_loss)
            logs['loss'].append(loss)

        logs.update({key: np.array(val) for key, val in logs.items()})
        self.rollback_states()
        return logs

    def cache_states(self):
        cache_dict = {'model_state': deepcopy(self.model.state_dict()),
                      'optimizer_state': deepcopy(self.optimizer.state_dict())}

        return cache_dict

    def rollback_states(self):
        self.model.load_state_dict(self.cache['model_state'])
        self.optimizer.load_state_dict(self.cache['optimizer_state'])