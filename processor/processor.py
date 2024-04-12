import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import logging
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils import Saver


class Processor:
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


    def set_optimizers(self, optimizer, optimizer_center):
        self.optimizer = optimizer
        self.optimizer_center = optimizer_center

    def set_loss_funcs(self, loss_func, center_criterion):
        self.loss_func = loss_func
        self.center_criterion = center_criterion

    def init_meters(self):
        self.cls_loss_meter = AverageMeter()
        self.tri_loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
    
    def reset_meters(self):
        self.cls_loss_meter.reset()
        self.tri_loss_meter.reset()
        self.acc_meter.reset()

    def log_info(self, n_iter):
        info_str = f"Epoch[{self.current_epoch}] Iteration[{(n_iter + 1)}/{len(self.train_loader)}]"
        info_str += f"loss: {self.loss_meter.avg:.3f}, "
        # info_str += f"Acc: {self.acc_meter.avg:.3f}, Base Lr: {self.scheduler._get_lr(self.current_epoch)[0]:.2e}"
        info_str += f"Acc: {self.acc_meter.avg:.3f}, Base Lr: {self.optimizer.param_groups[0]['lr']:.2e}"
        self.logger.info(info_str)
    
    def dump_metrics_data_to_tensorboard(self):
        self.saver.dump_metric_tb(self.cls_loss_meter.avg, self.current_epoch, f'losses', f'cls_loss')
        self.saver.dump_metric_tb(self.tri_loss_meter.avg, self.current_epoch, f'losses', f'tri_loss')
        self.saver.dump_metric_tb(self.loss_meter.avg, self.current_epoch, f'losses', f'loss')        
        self.saver.dump_metric_tb(self.acc_meter.avg, self.current_epoch, f'losses', f'acc')
        self.saver.dump_metric_tb(self.optimizer.param_groups[0]['lr'], self.current_epoch, f'losses', f'lr')

    def log_epoch_stats(self, epoch, time_per_batch):
        
        if self.isVideo:
            num_samples = self.cfg.DATALOADER.P * self.cfg.DATALOADER.K * self.cfg.DATALOADER.NUM_TRAIN_IMAGES
            self.logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {num_samples/time_per_batch:.1f}[samples/s]")
        else:
            self.logger.info(f"Epoch {epoch} done. Time per batch: {time_per_batch:.3f}[s] Speed: {self.train_loader.batch_size/time_per_batch:.1f}[samples/s]")

    def train_step(self):
        
        self.model.train()

        for n_iter, (img, pid, target_cam, target_view) in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()
            self.optimizer_center.zero_grad()

            img = img.to(self.device)
            target = pid.to(self.device)
            target_cam = target_cam.to(self.device) if self.cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(self.device) if self.cfg.MODEL.SIE_VIEW else None

            # score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view)
            # ID_LOSS, TRI_LOSS = self.loss_func(score, feat, target, target_cam)
            # loss = self.cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + self.cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            with amp.autocast(enabled=False):
                score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view)
                loss = self.loss_func(score, feat, target, target_cam)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            # self.cls_loss_meter.update(ID_LOSS.item(), img.shape[0])
            # self.tri_loss_meter.update(TRI_LOSS.item(), img.shape[0])

            self.loss_meter.update(loss.item(), img.shape[0])
            self.acc_meter.update(acc.item(), 1)

            self.acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % self.log_period == 0:
                self.log_info(n_iter)
        
        return n_iter
            
            
    def test_step(self):
        self.model.eval()
        for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(self.val_loader):
            
            with torch.no_grad():
                img = img.to(self.device)
                camids = camids.to(self.device)
                target_view = target_view.to(self.device)
                feat = self.model(img, cam_label=camids, view_label=target_view)
                self.evaluator.update((feat, pid, camid))
        
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        self.logger.info("Validation Results - Epoch: {}".format(self.current_epoch))
        self.logger.info("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()
        
        self.evaluator.reset()

        return cmc, mAP


    def train(self):
        checkpoint_period = self.cfg.SOLVER.CHECKPOINT_PERIOD
        eval_period = self.cfg.SOLVER.EVAL_PERIOD
        
        self.scaler = amp.GradScaler()
        freeze_layers = ['base', 'pyramid_layer']

        self.model.to(self.device)

        # TODO Remove later
        self.epochs += self.start_epoch
        
        for epoch in range(self.start_epoch+1, self.epochs+1):
            self.current_epoch = epoch
            start_time = time.time()
            
            self.reset_meters()
            
            # self.scheduler.step(epoch)

            # -----------------
            # The convolution layer and transformer layers are frozen in the first five epochs 
            # to train the classifier layers. 
            # After these five epochs, the whole network is trained.
            # WARMUP_EPOCHS = 5 in config
            # -----------------
            if epoch < self.cfg.SOLVER.WARMUP_EPOCHS:  # freeze layers for 2000 iterations
                    for name, module in self.model.named_children():
                        if name in freeze_layers:
                            module.eval()
            
            batch_num = self.train_step()

            self.dump_metrics_data_to_tensorboard()

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (batch_num + 1)

            self.log_epoch_stats(epoch, time_per_batch)

            if epoch % checkpoint_period == 0:
                self.saver.save_model_for_resume(self.model, 
                                                os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.NAME + '_resume_{}.pth'.format(epoch)), 
                                                epoch, 
                                                self.optimizer,
                                                self.scheduler)                
            if epoch % eval_period == 0 or epoch == 1:
                self.evaluator.reset()
                cmc, mAP = self.test_step()
            
                self.saver.dump_metric_tb(mAP, epoch, f'v2v', f'mAP')
                for cmc_v in [1, 5, 10, 20]:
                    self.saver.dump_metric_tb(cmc[cmc_v-1], epoch, f'v2v', f'cmc{cmc_v}')

        return cmc, mAP
    
    def inference(self):
        
        self.evaluator.reset()
        self.model.to(self.device)

        self.model.eval()

        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(self.val_loader):
            with torch.no_grad():
                img = img.to(self.device)
                
                camids = camids.to(self.device) if self.cfg.MODEL.SIE_CAMERA else None
                target_view = target_view.to(self.device) if self.cfg.MODEL.SIE_VIEW else None               
                
                feat = self.model(img, cam_label=camids, view_label=target_view)
                self.evaluator.update((feat, pid, camid))

        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        print("Inference Results ")
        print("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            print("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))


