import torch
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from processor.processor_mobilenet import ProcessorMobileNet
from processor.make_optimizer import make_optimizer
from processor import save_model
from loss.build_loss import make_loss
from utils.display import plot_loss_curves
from utils.logger import setup_logger
from solver import create_scheduler
from utils import Saver, setup_logger
from utils.misc import set_seeds
from models.resnet_orig import ResNetModel
from models.mobilenetV2 import MobileNetV2
from loss.softmax_loss import CrossEntropyLabelSmooth


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    set_seeds()
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/mobilenet.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger("CheckpointCharlie.train_mobilenet", output_dir, if_train=True)
    logger.info("Using {} as config file".format(args.config_file))
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    for i, (imgs, pids, _, _) in enumerate(train_loader):
        print(imgs.shape)
        print(pids)
        break

    cfg.NUM_CLASSES = num_classes
    # model = BuildModel(camera_num, view_num, cfg)
    # model = ResNetModel(device=device)
    model = MobileNetV2(cfg.NUM_CLASSES)

    x = model(imgs[0].unsqueeze(0))
    print(x)

    summary(model=model, 
        input_size=(32, 3, 256, 128),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

    # exit(0)
    loss_func = CrossEntropyLabelSmooth(num_classes=num_classes, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # loss_func, center_criterion = make_loss(cfg, num_classes=num_classes, device=device)
    # optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    # scheduler = create_scheduler(cfg, optimizer)

    current_epoch = 0
    if cfg.MODEL.PRETRAIN_CHOICE == 'resume':
            model_path = cfg.MODEL.PRETRAIN_PATH        
            print('Loading pretrained model for resume from {}'.format(model_path))
            model, optimizer, current_epoch, scheduler, _ = model.load_param_resume(model_path, optimizer, exp_lr_scheduler)

    start_time = time.perf_counter()

    proc = ProcessorMobileNet(cfg, 
                     model,
                     num_query,
                     train_loader,
                     val_loader,
                     start_epoch=current_epoch,                      
                     device=device)
    proc.set_optimizers(optimizer, None)
    proc.set_loss_funcs(loss_func, None)
    proc.scheduler = exp_lr_scheduler
    proc.logger = logger

    cmc, mAP = proc.train()
    logger.info("mAP: {:.3%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))

    # saver = Saver(output_dir, f'tensorboard')    

