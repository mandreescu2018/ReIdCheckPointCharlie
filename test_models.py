import torch
import argparse
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel, Bottleneck
from torchinfo import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )

    model = Bottleneck(inplanes=64,
                       planes=751)
    
    print(model)

    # torch.Size([32, 64, 64, 32])

    summary(model=model,
            input_size=(32, 64, 64, 32),
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            col_width=20,
            row_settings=['var_names'])
