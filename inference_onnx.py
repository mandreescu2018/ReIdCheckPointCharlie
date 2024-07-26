import numpy as np
import torch
import onnxruntime as rt
import onnx
from torchinfo import summary
import argparse
import time
from config import cfg
from datasets.make_dataloader import make_dataloader
from models.simple_model import BuildModel
from models.model_selector import get_model
from processor.engine_reid import do_inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print(device)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configurations/main.yml", help="path to config file", type=str
    )
    
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    cfg.NUM_CLASSES = num_classes
    pytorch_model = BuildModel(camera_num, view_num, cfg)
    
    checkpoint = torch.load(cfg.TEST.WEIGHT)
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])

    onnx_model = onnx.load("Export/model.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = rt.InferenceSession("Export/model.onnx")

    i = 0

    pytorch_model.to(device)
    pytorch_model.eval()
    

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        if i>2:
            break    
        with torch.no_grad():
            img = img.to(device)
            
            camids = None
            target_view = None               
            
            feat = pytorch_model(img, cam_label=camids, view_label=target_view)
            print("pytorch feat:", feat)

            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}

            ort_outs = ort_session.run(None, ort_inputs)
            print("ort_outs:", ort_outs)

            # compute ONNX Runtime output prediction
            # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
            # ort_outs = ort_session.run(None, ort_inputs)

            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(to_numpy(feat), ort_outs[0], rtol=1e-03, atol=1e-05)
            # np.testing.assert_allclose(feat.cpu(), ort_outs[0], rtol=1e-03, atol=1e-05)
            print("Exported model has been tested with ONNXRuntime, and the result looks good!")
            
            i += 1
                
                # self.evaluator.update((feat, pid, camid))
    # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
    #     img = img.to(device)
    #     feat = pytorch_model(img)
    #     print(feat)
    # cfg.NUM_CLASSES = num_classes
    # model = BuildModel(camera_num, view_num, cfg)
    
    # checkpoint = torch.load(cfg.TEST.WEIGHT)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # summary(model=model, 
    #         input_size=(32, 3, 256, 128),
    #         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
    #         col_width=20,
    #         row_settings=['var_names'])
    
    # proc = Processor(cfg, 
    #                  model,
    #                  num_query,
    #                  train_loader,
    #                  val_loader,
    #                  device=device)
    # proc.inference()