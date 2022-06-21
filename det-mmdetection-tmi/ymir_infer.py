from mmdet.apis import init_detector, inference_detector
from easydict import EasyDict as edict
from mmdet.utils.util_ymir import get_weight_file

class YmirModel:
    def __init__(self, cfg:edict):
        self.cfg = cfg

        # Specify the path to model config and checkpoint file
        config_file = cfg.param.config_file
        checkpoint_file = get_weight_file(cfg)

        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def infer(self, img):
        return inference_detector(self.model, img)
