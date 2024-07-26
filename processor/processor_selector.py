from .processor_hacnn import ProcessorHaCnn
from .processor_mobilenet import ProcessorMobileNet

__factory = {
    'hacnn': ProcessorHaCnn,
    'mobilenet': ProcessorMobileNet,
}

def get_processor(cfg):
    proc = __factory[cfg.MODEL.NAME]
    return proc