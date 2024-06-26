import torch
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel, ModifiedResNet
from models.resnet_orig import ResNetModel
from processor.engine_reid import train
from processor.make_optimizer import make_optimizer
from processor import save_model
from loss.build_loss import make_loss
from utils.display import plot_loss_curves
from utils.logger import setup_logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
    w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
    
    cfg.NUM_CLASSES = num_classes
    # model = ResNetModel(device=device)
    model = ModifiedResNet(layers=(3, 4, 6, 3),
                output_dim=1024,
                heads=32,
                input_resolution=h_resolution*w_resolution,
                width=64)


    summary(model=model, 
            input_size=(32, 3, 256, 128),
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            col_width=20,
            row_settings=['var_names'])
    
    # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
    #     img = img.to(device)
    #     feat = model(img)
    #     print(feat)
    #     break
