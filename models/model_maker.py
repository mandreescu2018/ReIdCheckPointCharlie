from .harmonious_attention_cnn import HACNN
from .resnet_orig import ResNetModel

__factory = {
    'resnet50': ResNetModel,
    'hacnn': HACNN,
}

def get_model(cfg):
    if cfg.MODEL.NAME == 'hacnn':
        model = HACNN(cfg.NUM_CLASSES)
        return model
    elif cfg.MODEL.NAME not in __factory.keys():
        raise KeyError(f"Unknown model: {cfg.MODEL.NAME}")
    
