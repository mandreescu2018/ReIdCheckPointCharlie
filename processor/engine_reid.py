import logging
import torch
from typing import List, Dict
import yacs.config
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval

def train_step(model: torch.nn.Module, 
               epoch: int,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               cfg,
               device: torch.device):
    # Put model in train mode
    
    model.train()

    log_period = cfg.SOLVER.LOG_PERIOD

    # Setup train loss and train accuracy values
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    train_loss, train_acc = 0, 0
    for n_iter, (img, pid, target_cam, target_view) in enumerate(dataloader):
       
        # Send data to target device
        img = img.to(device)
        target = pid.to(device)

        target_cam = target_cam.to(device) if cfg.MODEL.SIE_CAMERA else None
        target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None

        # 1. Forward pass
        score, feat = model(img, target, cam_label=target_cam, view_label=target_view)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(score, feat, target, target_cam)

        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == target).float().mean()
        else:
            acc = (score.max(1)[1] == target).float().mean()

        loss_meter.update(loss.item(), img.shape[0])
        acc_meter.update(acc.item(), 1)

        if (n_iter + 1) % log_period == 0:
            print(f"Epoch{epoch} Iteration: {n_iter + 1}/{len(dataloader)} Loss: {loss_meter.avg:.3f}, Acc: {acc_meter.avg:.3f}")
                
                # logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                #             .format(epoch, (n_iter + 1), len(dataloader),
                #                     loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
    
    # Adjust metrics to get average loss and accuracy per batch 
    # train_loss = loss_meter.avg
    # train_acc = acc_meter.avg
    return loss_meter.avg, acc_meter.avg
    # return train_loss, train_acc

def test_step_reid(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              cfg,
              num_query: int,
              epoch: int,
              device: torch.device):
   # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(dataloader):
        with torch.no_grad():
            img = img.to(device)
            
            camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None
            
            feat = model(img, cam_label=camids, view_label=target_view)
            # score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            # loss = loss_fn(score, feat, target, target_cam)

            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print("Validation Results - Epoch: {}".format(epoch))
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    # logger.info("Validation Results - Epoch: {}".format(epoch))
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return test_loss, test_acc
            

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              cfg,
              device: torch.device):
   # Put model in eval mode
    model.eval() 

    log_period = cfg.SOLVER.LOG_PERIOD

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        # Loop through data loader data batches
        for n_iter, (img, pid, camid, camids, target_view, _) in enumerate(dataloader):
            
            # Send data to target device
            img = img.to(device)
            target = pid.to(device)
            # camids = camids.to(device) if cfg.MODEL.SIE_CAMERA else None
            target_cam = camid.to(device) if cfg.MODEL.SIE_CAMERA else None

            target_view = target_view.to(device) if cfg.MODEL.SIE_VIEW else None

            # 1. Forward pass
            feat = model(img, target, cam_label=camid, view_label=target_view)
            
            score, feat = model(img, target, cam_label=target_cam, view_label=target_view)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(score, feat, target, target_cam)

            # Calculate and accumulate accuracy
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                print(f"Iteration: {n_iter + 1}/{len(dataloader)} Loss: {loss_meter.avg:.3f}, Acc: {acc_meter:.3f}")
        
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = loss_meter.avg
    test_acc = acc_meter.avg
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          cfg: yacs.config.CfgNode,
          num_query: int,
          device: torch.device) -> Dict[str, List]:
  
    """Trains and tests a PyTorch model."""
    
    # Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    model.to(device)

    for epoch in range(epochs):

        train_loss, train_acc = train_step(model=model,
                                            epoch=epoch,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            cfg=cfg,
                                            device=device)
        test_loss, test_acc = test_step_reid(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            cfg=cfg,
            num_query=num_query,
            epoch=epoch,
            device=device)
    
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

from PIL import Image
from torchvision import transforms

def load_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 128)),  # Assuming the model uses 224x224 inputs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def inference_one_pic(model: torch.nn.Module,                        
                        imgpath: str,
                        device: torch.device):
    model.eval()
    img = load_image(imgpath)
    with torch.no_grad():
        img = img.to(device)
        feat = model(img)
    return feat

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 device):
    
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    print("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    print("mAP: {:.3%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
        print("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
  

