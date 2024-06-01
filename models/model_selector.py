from .harmonious_attention_cnn import HACNN
from .mobilenetV2 import MobileNetV2

__factory = {
    'hacnn': HACNN,
    'mobilenet': MobileNetV2,
}

def get_model(cfg):
    model = __factory[cfg.MODEL.NAME](cfg.NUM_CLASSES)
    return model
