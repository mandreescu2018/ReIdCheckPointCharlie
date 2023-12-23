import torch
from torchvision.models import resnet50, ResNet50_Weights

class ResNetModel(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights = weights).to(device)
    
    def forward(self, x):
        return self.model(x)