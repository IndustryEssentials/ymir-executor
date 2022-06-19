from mmdet.apis import init_detector, inference_detector
from easydict import EasyDict as edict

class YmirModel:
    def __init__(self, cfg:edict):
        self.cfg = cfg

        # Specify the path to model config and checkpoint file
        config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

        # build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')

    def infer(self, img):
        return inference_detector(self.model, img)
