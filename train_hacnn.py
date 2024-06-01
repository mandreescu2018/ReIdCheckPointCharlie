import torch
from torch import nn
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from processor.processor_hacnn import ProcessorHaCnn as Processor
from processor.make_optimizer import make_optimizer
from processor import save_model
from loss.build_loss import make_loss
from utils.display import plot_loss_curves
from utils.logger import setup_logger
from solver import create_scheduler
from utils import Saver, setup_logger
from utils.misc import set_seeds
from models.model_maker import get_model
from models.mobilenetV2 import MobileNetV2
from loss.softmax_loss import CrossEntropyLabelSmooth


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seeds()
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/hacnn.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger("CheckpointCharlie.train_hacnn", output_dir, if_train=True)
    logger.info("Using {} as config file".format(args.config_file))
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.NUM_CLASSES = num_classes
    model = get_model(cfg)
    # model = ResNetModel(device=device)
    # model = MobileNetV2(cfg.NUM_CLASSES)

    summary(model=model, 
        input_size=(32, 3, 160, 64),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])
    
    # logger.info(str(model_stats))

    # exit(0)

    # ADAM[12] algorithm at the initial learning rate 5X10e-4 with the two moment terms beta1 = 0:9 and beta2 = 0:999.
    # We set the batch size to 32, epoch to 150, momentum to 0.9.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), weight_decay=5e-4)

    loss_func = nn.CrossEntropyLoss()
    
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # loss_func, center_criterion = make_loss(cfg, num_classes=num_classes, device=device)
    # optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    # scheduler = create_scheduler(cfg, optimizer)

    current_epoch = 0
    if cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            model_path = cfg.MODEL.PRETRAIN_PATH        
            print('Loading pretrained model for resume from {}'.format(model_path))
            model, optimizer, current_epoch, scheduler, _ = model.load_param_resume(model_path, optimizer, exp_lr_scheduler)

    start_time = time.perf_counter()

    proc = Processor(cfg, 
                     model,
                     num_query,
                     train_loader,
                     val_loader,
                     start_epoch=current_epoch,                      
                     device=device)
    proc.set_optimizers(optimizer)
    proc.set_loss_funcs(loss_func)
    proc.scheduler = exp_lr_scheduler
    proc.logger = logger

    cmc, mAP = proc.train()
    logger.info("mAP: {:.3%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))

    # saver = Saver(output_dir, f'tensorboard')    

