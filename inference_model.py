import torch
import onnxruntime as rt
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models import get_model
from processor.processor_mobilenet import Processor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/hacnn.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.NUM_CLASSES = num_classes
    model = get_model(cfg)
    
    checkpoint = torch.load(cfg.TEST.WEIGHT)
    model.load_state_dict(checkpoint['model_state_dict'])

    summary(model=model, 
            input_size=(32, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]),
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            col_width=20,
            row_settings=['var_names'])
    
    start = time.perf_counter()
    proc = Processor(cfg, 
                     model,
                     num_query,
                     train_loader,
                     val_loader,
                     device=device)
    proc.inference()
    end = time.perf_counter()

    print(f"Time taken to infer: {end-start:0.4f} seconds")