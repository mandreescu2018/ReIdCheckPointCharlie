import argparse
from config import cfg
from datasets.make_dataloader import make_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )
    

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    print(cfg.DATASETS.NAMES)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    print(len(train_loader))