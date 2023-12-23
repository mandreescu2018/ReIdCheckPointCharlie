import torch
from torch import nn
from utils.misc import set_seeds

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        # initialize trainable params
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        
        # The code computes the Euclidean distance between each embedding in x and the class centers. 
        # It uses matrix operations to do this efficiently.
        # It calculates the squared Euclidean distances between embeddings and centers and accumulates them in distmat.
        # distmat = torch.randn(16, 751).to(self.device)
        distmat = torch.sum(x ** 2, dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.sum(self.centers ** 2, dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # The result of this line of code is to compute the Euclidean distances between each sample in the batch (x) 
        # and each class center (self.centers) and store the distances in the distmat tensor. 
        # The choice of -2 in alpha is related to the way Euclidean distances are computed, 
        # which is the square root of the sum of squared differences. The -2 scaling is applied to match this calculation, 
        # as it effectively squares the differences. This is a common technique used to compute distances efficiently in a 
        # batch-wise manner within deep learning models.
        
        # Performs a matrix multiplication of the matrices x and self.centers.t(). The matrix input is added to the final result.
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        dist = dist.clamp(min=1e-12, max=1e+12)
        loss = dist.sum() / batch_size
        return loss




