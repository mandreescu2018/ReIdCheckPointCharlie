import torch
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from processor.processor import Processor
from processor.make_optimizer import make_optimizer
from processor import save_model
from loss.build_loss import make_loss
from utils.display import plot_loss_curves
from utils.logger import setup_logger
from solver import create_scheduler
from utils import Saver, setup_logger
from utils.misc import set_seeds
from models.resnet_orig import ResNetModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seeds()
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger("CheckpointCharlie.train", output_dir, if_train=True)
    logger.info("Using {} as config file".format(args.config_file))
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.NUM_CLASSES = num_classes
    model = BuildModel(camera_num, view_num, cfg)
    # model = ResNetModel(device=device)

    summary(model=model, 
        input_size=(32, 3, 256, 128),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes, device=device)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    current_epoch = 0
    if cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            model_path = cfg.MODEL.PRETRAIN_PATH        
            print('Loading pretrained model for resume from {}'.format(model_path))
            model, optimizer, current_epoch, scheduler, _ = model.load_param_resume(model_path, optimizer, scheduler)

    start_time = time.perf_counter()

    proc = Processor(cfg, 
                     model,
                     num_query,
                     train_loader,
                     val_loader,
                     start_epoch=current_epoch,                      
                     device=device)
    proc.set_optimizers(optimizer, optimizer_center)
    proc.set_loss_funcs(loss_func, center_criterion)
    proc.scheduler = scheduler
    proc.logger = logger

    cmc, mAP = proc.train()
    logger.info("mAP: {:.3%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))

    # saver = Saver(output_dir, f'tensorboard')    

