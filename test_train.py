import torch
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from processor.engine_reid import train
from processor.make_optimizer import make_optimizer
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

    
    model = BuildModel(num_classes, camera_num, view_num, cfg)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes, device=device)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    start_time = time.perf_counter()

    res = train(model=model,
          train_dataloader=train_loader,
          test_dataloader=val_loader,
          optimizer=optimizer,
          loss_fn=loss_func,
          epochs=cfg.SOLVER.MAX_EPOCHS,
          cfg=cfg,
          device=device)
    
    end_time = time.perf_counter()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    plot_loss_curves(res)

    

