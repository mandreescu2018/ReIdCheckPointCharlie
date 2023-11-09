import torch
from utils.misc import set_seeds
from loss.center_loss import CenterLoss

if __name__ == '__main__':
    set_seeds()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    center_loss = CenterLoss(device=device)    
    features = torch.rand(16, 2048).to(device)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long().to(device)
        
    loss = center_loss(features, targets)
    print(loss)
