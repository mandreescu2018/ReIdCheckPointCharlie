import torch
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from models.mobilenetV2 import MobileNetV2
from processor.engine_reid import do_inference
from processor.processor import Processor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "Export/market_model.onnx"

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/mobilenet.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    dummy_input = torch.randn(None, 3, 256, 128, device='cuda')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.NUM_CLASSES = num_classes
    # model = BuildModel(camera_num, view_num, cfg)
    model = MobileNetV2(num_classes=num_classes)
    
    checkpoint = torch.load(cfg.TEST.WEIGHT)
    model.load_state_dict(checkpoint['model_state_dict'])

    summary(model=model, 
            input_size=(1, 3, 256, 128),
            col_names=['input_size', 'output_size', 'num_params', 'trainable'],
            col_width=20,
            row_settings=['var_names'])
    
    # torch.onnx.export(
    #     model,
    #     sample_input, 
    #     onnx_model_path,
    #     verbose=False,
    #     input_names=['input'],
    #     output_names=['output'],
    #     opset_version=12
    # )
    torch.onnx.export(model,                 # model being run
                  dummy_input,           # model input (or a tuple for multiple inputs)
                  model_path,          # where to save the model (can be a file or file-like object)
                  export_params=True,    # store the trained parameter weights inside the model file
                  opset_version=12,      # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],     # the model's input names
                  output_names=['output'],   # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})
    
    