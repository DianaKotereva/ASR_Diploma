import torch
from torch import nn
from utils import ConstrativeLoss, sample_negatives
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import optim
from transformers_f.src.transformers.activations import ACT2FN
from transformers_f.src.transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from modeling_segmentation import Wav2Vec2ModelForSegmentation



class SegmentTransformer(LightningModule):
    
    def __init__(self, cfg):
        super(SegmentTransformer, self).__init__()
        self.cfg = cfg
        self.wav2vec_segm = Wav2Vec2ModelForSegmentation.from_pretrained(cfg.model_path)
        self.hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.learning_rate = self.cfg.learning_rate

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.cfg.optimizer == "sgd":
            self.opt = optim.SGD(parameters, lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif self.cfg.optimizer == "adam":
            self.opt = optim.Adam(parameters, lr=self.learning_rate, weight_decay=5e-4)
        elif self.cfg.optimizer == "ranger":
            self.opt = optim_extra.Ranger(parameters, lr=self.learning_rate, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), 
                                          eps=1e-5,
                                          weight_decay=0)
        else:
            raise Exception("unknown optimizer")
        print(f"optimizer: {self.opt}")
        self.scheduler = optim.lr_scheduler.StepLR(self.opt,
                                                   step_size=self.cfg.lr_anneal_step,
                                                   gamma=self.cfg.lr_anneal_gamma)
        
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
#                                                       patience=1, threshold=0.0001, threshold_mode='rel',
#                                                       cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        return [self.opt]

    def training_step(self, batch, batch_idx):
        
        loss, res_dict = self.wav2vec_segm.compute_all(batch['batch'].to(self.device), batch['boundaries'], 
                                                       num_epoch = self.trainer.current_epoch, 
                                                       attention_mask = batch['attention_mask'].to(self.device))
        self.log("train_loss", loss)
        for key, value in res_dict.items():
            self.log(key, value)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss, res_dict = self.wav2vec_segm.compute_all(batch['batch'].to(self.device), batch['boundaries'], 
                                                       num_epoch = self.trainer.current_epoch, 
                                                       attention_mask = batch['attention_mask'].to(self.device))
        self.log("val_loss", loss)
        for key, value in res_dict.items():
            self.log('val_'+key, value)
        return loss