import torchvision.transforms as T
import numpy as np
import math
from PIL import Image

class Transforms:
    def __init__(self, cfg):
        # self.cfg = cfg
        
        self.train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            T.RandomErasing(p=cfg.INPUT.PROB)
        ])
        

        self.test_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

        mean, var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        self.video_train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.Pad(10),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, var),
            # T.RandomErasing(p=0.5, scale=erase_scale, ratio=erase_ratio)

            ])
        self.video_test_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean, var)
        ])
        

    def get_train_transforms(self):
        return self.train_transforms

    def get_test_transforms(self):
        return self.test_transforms