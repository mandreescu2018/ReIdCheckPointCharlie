import torch
import torch.nn as nn
import torch.optim as optim
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .softmax_loss import CrossEntropyLabelSmooth

class OverAllLoss(nn.Module):
    def __init__(self, cfg) -> None:
        super(OverAllLoss, self).__init__()
        self.cfg = cfg
        self.compose()
    
    def compose(self):
        
        # Triplet loss 
        margin = None if self.cfg.MODEL.NO_MARGIN else self.cfg.SOLVER.MARGIN
        self.triplet = TripletLoss(margin)

        # Center loss
        self.center_loss = CenterLoss(
            num_classes=self.cfg.NUM_CLASSES, feat_dim=self.cfg.MODEL.BACKBONE_EMB_SIZE
        )

        self.xent = CrossEntropyLabelSmooth(num_classes=self.cfg.NUM_CLASSES)

    # score, feat, target, target_cam
    def forward(self, score, features, targets, target_cam):
        # Calculate individual losses
        triplet_loss_value = self.triplet(features, targets)
        triplet_loss_value *= self.cfg.SOLVER.QUERY_CONTRASTIVE_WEIGHT

        center_loss_value = self.center_loss(features, targets)
        center_loss_value *= self.cfg.SOLVER.CENTER_LOSS_WEIGHT

        xent_loss_value = self.xent(score, targets) 
        xent_loss_value *= self.cfg.SOLVER.QUERY_XENT_WEIGHT

        total_loss_value = triplet_loss_value + center_loss_value + xent_loss_value

        return total_loss_value


class ComposedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComposedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target_regression, target_classification):
        # Calculate individual losses
        mse_loss_value = self.mse_loss(prediction[:, :4], target_regression)
        cross_entropy_loss_value = self.cross_entropy_loss(prediction[:, 4:], target_classification)

        # Combine losses (you can adjust the weights using alpha)
        composed_loss = self.alpha * mse_loss_value + (1 - self.alpha) * cross_entropy_loss_value

        return composed_loss
    
if __name__ == "___main__":
    # Example usage:
    # Assuming prediction is a tensor of shape (batch_size, 8),
    # where the first 4 values are regression predictions, and the next 4 values are classification predictions.
    # Assuming target_regression and target_classification are ground truth tensors.
    prediction = torch.randn(16, 8)
    target_regression = torch.randn(16, 4)
    target_classification = torch.randint(0, 2, (16,))

    composed_loss_fn = ComposedLoss(alpha=0.7)
    loss = composed_loss_fn(prediction, target_regression, target_classification)
    print(loss)