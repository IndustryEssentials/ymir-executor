import argparse
from typing import Any, List

import numpy as np
from easydict import EasyDict as edict
from mmcv import DictAction
from nptyping import NDArray, Shape
from ymir_exc import result_writer as rw

from mmdet.apis import inference_detector, init_detector
from mmdet.utils.util_ymir import get_weight_file

DETECTION_RESULT = NDArray[Shape['*,5'], Any]


def parse_option(cfg_options: str) -> dict:
    parser = argparse.ArgumentParser(description='parse cfg options')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args(f'--cfg-options {cfg_options}'.split())
    return args.cfg_options


def mmdet_result_to_ymir(results: List[DETECTION_RESULT],
                         class_names: List[str]) -> List[rw.Annotation]:
    ann_list = []
    for idx, result in enumerate(results):
        for line in result:
            if any(np.isinf(line)):
                continue
            x1, y1, x2, y2, score = line
            ann = rw.Annotation(
                class_name=class_names[idx],
                score=score,
                box=rw.Box(x=round(x1),
                           y=round(y1),
                           w=round(x2-x1),
                           h=round(y2-y1))
            )
            ann_list.append(ann)
    return ann_list


class YmirModel:
    def __init__(self, cfg: edict):
        self.cfg = cfg

        # Specify the path to model config and checkpoint file
        config_file = cfg.param.config_file
        checkpoint_file = get_weight_file(cfg)
        cfg_options = parse_option(
            cfg.param.cfg_options) if cfg.param.cfg_options else None

        # current infer can only use one gpu!!!
        gpu_ids = cfg.param.gpu_id
        gpu_id = gpu_ids.split(',')[0]
        # build the model from a config file and a checkpoint file
        self.model = init_detector(
            config_file, checkpoint_file, device=f'cuda:{gpu_id}', cfg_options=cfg_options)

    def infer(self, img):
        return inference_detector(self.model, img)
