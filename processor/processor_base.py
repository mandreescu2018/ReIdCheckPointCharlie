import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import logging
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils import Saver


class ProcessorBase:
    def __init__(self, cfg,                   
                 model,                 
                 num_query, 
                 train_loader,
                 val_loader, 
                 start_epoch = 0,
                 device='cuda'):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.epochs = cfg.SOLVER.MAX_EPOCHS
        self.start_epoch = start_epoch
        self.current_epoch = None
        self._logger = None
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = amp.GradScaler()
        self.isVideo = True if cfg.DATASETS.NAMES in ['mars', 'duke-video-reid', 'ilids', 'prid'] else False
        self._scheduler = None
        self._saver = None
        self.init_meters()
    
    @property
    def scheduler(self):
        """The scheduler property."""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        if not value:
            raise ValueError("Scheduler empty.")
        self._scheduler = value

    def init_meters(self):
        self.acc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
    
    def reset_meters(self):
        self.acc_meter.reset()
        self.loss_meter.reset()
    
    def dump_metrics_data_to_tensorboard(self):
        self.saver.dump_metric_tb(self.loss_meter.avg, self.current_epoch, f'losses', f'loss')        
        self.saver.dump_metric_tb(self.acc_meter.avg, self.current_epoch, f'losses', f'acc')
        self.saver.dump_metric_tb(self.optimizer.param_groups[0]['lr'], self.current_epoch, f'losses', f'lr')
    
    def log_info(self, n_iter):
        info_str = f"Epoch[{self.current_epoch}] Iteration[{(n_iter + 1)}/{len(self.train_loader)}]"
        info_str += f"loss: {self.loss_meter.avg:.3f}, "
        # info_str += f"Acc: {self.acc_meter.avg:.3f}, Base Lr: {self.scheduler._get_lr(self.current_epoch)[0]:.2e}"
        info_str += f"Acc: {self.acc_meter.avg:.3f}, Base Lr: {self.optimizer.param_groups[0]['lr']:.2e}"
        self.logger.info(info_str)

    # Tensorboard writer
    @property
    def saver(self):
        if self._saver is None:
            self._saver = Saver(self.cfg.OUTPUT_DIR, f'tensorboard')
        return self._saver

    @property
    def logger(self):        
        return self._logger
    
    @logger.setter
    def logger(self, value):
        if not value:
            raise ValueError("Logger empty.")
        self._logger = value

    def log_epoch_stats(self, epoch, time_per_batch):        
        if self.isVideo:
            num_samples = self.cfg.DATALOADER.P * self.cfg.DATALOADER.K * self.cfg.DATALOADER.NUM_TRAIN_IMAGES
            self.logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {num_samples/time_per_batch:.1f}[samples/s]")
        else:
            self.logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {self.train_loader.batch_size/time_per_batch:.1f}[samples/s]")

    def train(self):
        self.checkpoint_period = self.cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = self.cfg.SOLVER.EVAL_PERIOD
        self.model.to(self.device)


    def train_step(self):
        self.model.train()

    def test_step(self):
        self.model.eval()

    def inference(self):
        self.evaluator.reset()
        self.model.to(self.device)
        self.model.eval()

    def inference(self):        
        self.evaluator.reset()
        self.model.to(self.device)
        self.model.eval()

        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(self.val_loader):
            with torch.no_grad():
                img = img.to(self.device)                
                outputs = self.model(img)
                self.evaluator.update((outputs, pid, camid))

        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        print("Inference Results ")
        print("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            print("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))