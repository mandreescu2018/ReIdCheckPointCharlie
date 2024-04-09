import argparse
from config import cfg
from datasets import make_dataloader
import time
import matplotlib.pyplot as plt

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
    # print("len(train_loader)", len(train_loader))
    print("len(train_loader.dataset)", len(train_loader.dataset))
    print("len(val_loader.dataset)", len(val_loader.dataset))

    start_time = time.perf_counter()

    for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
        if n_iter < 5:
            print(f'Image: {img.shape}, Vid: {vid.shape}, targetcam: {target_cam.shape}, target_view: {target_view.shape}')
            # plt.imshow(img[0].permute(1,2,0))
            # plt.show()
        else:
            break
            
    
    print()
    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
    # for n_iter, (img, vid, target_cam, target_view) in enumerate(val_loader):
        if n_iter < 5:
            print(f'Image: {img.shape}, camids: {camids.shape}, target_view: {target_view.shape}')
            plt.imshow(img[0].permute(1,2,0))
            plt.show()

        else:
            break
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Elapsed time for loops: {elapsed_time:.6f} seconds")

