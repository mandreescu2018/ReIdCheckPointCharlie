import argparse
import torch
from config import cfg
from datasets.make_dataloader import make_dataloader
from processor.engine_reid_centr import train
from models.simple_model import BuildModel
from loss.composed_loss import OverAllLoss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/centroids.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    print(len(train_loader.dataset))
    cfg.NUM_CLASSES = num_classes
    loss_fn = OverAllLoss(cfg)

    model = BuildModel(camera_num=camera_num,
                       view_num=view_num,
                       cfg=cfg)
    
    print(model)
    

