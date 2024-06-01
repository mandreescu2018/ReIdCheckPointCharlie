import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import logging
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils import Saver
from.processor_base import ProcessorBase


class ProcessorMobileNet(ProcessorBase):
    def __init__(self, cfg, 
                    model,
                    num_query,
                    train_loader,
                    val_loader,
                    start_epoch=0,
                    device='cuda'):
        super(ProcessorMobileNet, self).__init__(cfg,
                                        model,
                                        num_query,
                                        train_loader,
                                        val_loader,
                                        start_epoch,
                                        device)
    

    def set_optimizers(self, optimizer, optimizer_center):
        self.optimizer = optimizer
        self.optimizer_center = optimizer_center

    def set_loss_funcs(self, loss_func, center_criterion):
        self.loss_func = loss_func
        self.center_criterion = center_criterion

    def init_meters(self):
        super(ProcessorMobileNet, self).init_meters()
        self.cls_loss_meter = AverageMeter()
        self.tri_loss_meter = AverageMeter()
    
    def reset_meters(self):
        super(ProcessorMobileNet, self).reset_meters()
        self.cls_loss_meter.reset()
        self.tri_loss_meter.reset()
    
    def train_step(self):
        
        # self.model.train()

        for n_iter, (img, pid, _, _) in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()
            

            img = img.to(self.device)
            label = pid.to(self.device)

            with amp.autocast(enabled=False):
                outputs = self.model(img)
                loss = self.loss_func(outputs, label)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc = (outputs.max(1)[1] == label).float().mean()

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
                outputs = self.model(img)
                self.evaluator.update((outputs, pid, camid))
        
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()
        self.logger.info("Validation Results - Epoch: {}".format(self.current_epoch))
        self.logger.info("mAP: {:.3%}".format(mAP))
        for r in [1, 5, 10, 20]:
            self.logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
        torch.cuda.empty_cache()
        
        self.evaluator.reset()

        return cmc, mAP


    def train(self):
        super(ProcessorMobileNet, self).train()
        
        self.scaler = amp.GradScaler()
        freeze_layers = ['base', 'pyramid_layer']

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

            if epoch % self.checkpoint_period == 0:
                self.saver.save_model_for_resume(self.model, 
                                                os.path.join(self.cfg.OUTPUT_DIR, self.cfg.MODEL.NAME + '_resume_{}.pth'.format(epoch)), 
                                                epoch, 
                                                self.optimizer,
                                                self.scheduler)                
            if epoch % self.eval_period == 0 or epoch == 1:
                self.evaluator.reset()
                cmc, mAP = self.test_step()
            
                self.saver.dump_metric_tb(mAP, epoch, f'v2v', f'mAP')
                for cmc_v in [1, 5, 10, 20]:
                    self.saver.dump_metric_tb(cmc[cmc_v-1], epoch, f'v2v', f'cmc{cmc_v}')

        return cmc, mAP
        

