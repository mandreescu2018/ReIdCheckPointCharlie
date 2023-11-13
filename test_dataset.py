import argparse
from config import cfg
from datasets.make_dataloader import make_dataloader
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )
    
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    print(cfg.DATASETS.NAMES)
    
    start_time = time.perf_counter()

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    end_time = time.perf_counter()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    # print(len(train_loader))

    for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
        print(f'Image: {img.shape}, Vid: {vid.shape}, targetcam: {target_cam.shape}, target_view: {target_view.shape}')

