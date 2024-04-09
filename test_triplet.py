import torch

from utils.misc import set_seeds
from loss.triplet_loss import TripletLoss
from scipy.io import loadmat

if __name__ == "__main__":
    set_seeds()

    

    features = torch.randn(128, 2048)

    # stacked_tensor = torch.stack((anchor, positive, negative), dim=0)

    # class_names_tensor = torch.randint(0, 200, size=(1, 128), dtype=torch.int32).squeeze()
    class_names_tensor = torch.tensor([394, 394, 394, 394, 430, 430, 430, 430,  41,  41,  41,  41, 265, 265,
        265, 265, 523, 523, 523, 523, 497, 497, 497, 497, 414, 414, 414, 414,
        310, 310, 310, 310, 488, 488, 488, 488, 366, 366, 366, 366, 597, 597,
        597, 597, 223, 223, 223, 223, 516, 516, 516, 516, 142, 142, 142, 142,
        288, 288, 288, 288, 143, 143, 143, 143,  97,  97,  97,  97, 633, 633,
        633, 633, 256, 256, 256, 256, 545, 545, 545, 545, 722, 722, 722, 722,
        616, 616, 616, 616, 150, 150, 150, 150, 317, 317, 317, 317, 101, 101,
        101, 101, 747, 747, 747, 747,  75,  75,  75,  75, 700, 700, 700, 700,
        338, 338, 338, 338, 483, 483, 483, 483, 573, 573, 573, 573, 103, 103,
        103, 103])
    triplet = TripletLoss(margin=1)
    loss_t, _, _ = triplet(features, class_names_tensor)
    print("loss_t", loss_t)

    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    # output = triplet_loss(anchor, positive, negative)
    # print(output)
