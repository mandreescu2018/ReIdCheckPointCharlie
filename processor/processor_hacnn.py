import os
import time
import torch
from torch.cuda import amp
from .processor_base import ProcessorBase


class ProcessorHaCnn(ProcessorBase):
    def __init__(self, cfg,
                 model,
                 num_query,
                 train_loader,
                 val_loader,
                 start_epoch=0,
                 device='cuda'):
        super(ProcessorHaCnn, self).__init__(cfg,
                                             model,
                                             num_query,
                                             train_loader,
                                             val_loader,
                                             start_epoch,
                                             device)
        self.init_meters()
    
    def set_optimizers(self, optimizer):
        self.optimizer = optimizer
        # self.optimizer_center = optimizer_center

    def set_loss_funcs(self, loss_func):
        self.loss_func = loss_func
        # self.center_criterion = center_criterion

    def train(self):
        super(ProcessorHaCnn, self).train()

        for epoch in range(self.start_epoch+1, self.epochs+1):
            self.current_epoch = epoch
            start_time = time.time()
            
            self.reset_meters()
            self.scheduler.step(epoch)

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
    
    def train_step(self):
        super().train_step()

        for n_iter, (img, pid, _, _) in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()            

            img = img.to(self.device)
            label = pid.to(self.device)

            with amp.autocast(enabled=False):
                outputs = self.model(img)
                if isinstance(outputs, tuple):
                    loss = self.loss_func(outputs[0], label)
                    for output in outputs[1:]:
                        loss += self.loss_func(output, label)
                else:
                    loss = self.loss_func(outputs, label)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc = (outputs[0].max(1)[1] == label).float().mean()

            self.loss_meter.update(loss.item(), img.shape[0])
            self.acc_meter.update(acc.item(), 1)

            self.acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % self.log_period == 0:
                self.log_info(n_iter)
        
        return n_iter
    
    def test_step(self):
        super().test_step()

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

# Path: processor/processor_base.py
    



    