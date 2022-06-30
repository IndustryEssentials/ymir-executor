"""
utils function for ymir and yolov5
"""
import glob
import os.path as osp
import shutil
from enum import IntEnum
from typing import Any, List, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from nptyping import NDArray, Shape, UInt8
from ymir_exc import env
from ymir_exc import result_writer as rw

from models.common import DetectMultiBackend
from models.experimental import attempt_download
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2    # training/mining/infer
    POSTPROCESS = 3  # export model


BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_ymir_process(stage: YmirStage, p: float) -> float:
    # const value for ymir process
    PREPROCESS_PERCENT = 0.1
    TASK_PERCENT = 0.8
    POSTPROCESS_PERCENT = 0.1

    if p < 0 or p > 1.0:
        raise Exception(f'p not in [0,1], p={p}')

    if stage == YmirStage.PREPROCESS:
        return PREPROCESS_PERCENT * p
    elif stage == YmirStage.TASK:
        return PREPROCESS_PERCENT + TASK_PERCENT * p
    elif stage == YmirStage.POSTPROCESS:
        return PREPROCESS_PERCENT + TASK_PERCENT + POSTPROCESS_PERCENT * p
    else:
        raise NotImplementedError(f'unknown stage {stage}')


def get_merged_config() -> edict:
    """
    merge ymir_config and executor_config
    """
    merged_cfg = edict()
    # the hyperparameter information
    merged_cfg.param = env.get_executor_config()

    # the ymir path information
    merged_cfg.ymir = env.get_current_env()
    return merged_cfg


def get_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    """
    if cfg.ymir.run_training:
        model_params_path = cfg.param.get('pretrained_model_params',[])
    else:
        model_params_path = cfg.param.model_params_path

    model_dir = osp.join(cfg.ymir.input.root_dir,
                         cfg.ymir.input.models_dir)
    model_params_path = [p for p in model_params_path if osp.exists(osp.join(model_dir, p))]

    # choose weight file by priority, best.pt > xxx.pt
    if 'best.pt' in model_params_path:
        return osp.join(model_dir, 'best.pt')
    else:
        for f in model_params_path:
            if f.endswith('.pt'):
                return osp.join(model_dir, f)

    return ""


def download_weight_file(model_name):
    weights = attempt_download(f'{model_name}.pt')
    return weights


class YmirYolov5():
    """
    used for mining and inference to init detector and predict.
    """

    def __init__(self, cfg: edict):
        self.cfg = cfg
        device = select_device(cfg.param.get('gpu_id', 'cpu'))

        self.model = self.init_detector(device)
        self.device = device
        self.class_names = cfg.param.class_names
        self.stride = self.model.stride
        self.conf_thres = float(cfg.param.conf_thres)
        self.iou_thres = float(cfg.param.iou_thres)

        img_size = int(cfg.param.img_size)
        imgsz = (img_size, img_size)
        imgsz = check_img_size(imgsz, s=self.stride)

        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
        self.img_size = imgsz

    def init_detector(self, device: torch.device) -> DetectMultiBackend:
        weights = get_weight_file(self.cfg)

        data_yaml = osp.join(self.cfg.ymir.output.root_dir, 'data.yaml')
        model = DetectMultiBackend(weights=weights,
                                   device=device,
                                   dnn=False,  # not use opencv dnn for onnx inference
                                   data=data_yaml)  # dataset.yaml path

        return model

    def predict(self, img: CV_IMAGE) -> NDArray:
        """
        predict single image and return bbox information
        img: opencv BGR, uint8 format
        """
        # preprocess: padded resize
        img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

        # preprocess: convert data format
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)

        img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
        img1.unsqueeze_(dim=0)  # expand for batch dim
        pred = self.model(img1)

        # postprocess
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        classes = None  # not filter class_idx in results
        agnostic_nms = False
        max_det = 1000

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        result = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()
                result.append(det)

        # xyxy, conf, cls
        if len(result) > 0:
            tensor_result = torch.cat(result, dim=0)
            numpy_result = tensor_result.data.cpu().numpy()
        else:
            numpy_result = np.zeros(shape=(0, 6), dtype=np.float32)

        return numpy_result

    def infer(self, img: CV_IMAGE) -> List[rw.Annotation]:
        anns = []
        result = self.predict(img)

        for i in range(result.shape[0]):
            xmin, ymin, xmax, ymax, conf, cls = result[i, :6].tolist()
            ann = rw.Annotation(class_name=self.class_names[int(cls)], score=conf, box=rw.Box(
                x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))

            anns.append(ann)

        return anns


def convert_ymir_to_yolov5(cfg: edict) -> None:
    """
    convert ymir format dataset to yolov5 format
    generate data.yaml for training/mining/infer
    """

    data = dict(path=cfg.ymir.output.root_dir,
                nc=len(cfg.param.class_names),
                names=cfg.param.class_names)
    for split, prefix in zip(['train', 'val', 'test'], ['training', 'val', 'candidate']):
        src_file = getattr(cfg.ymir.input, f'{prefix}_index_file')
        if osp.exists(src_file):
            shutil.copy(src_file, f'{cfg.ymir.output.root_dir}/{split}.tsv')

        data[split] = f'{split}.tsv'

    with open(osp.join(cfg.ymir.output.root_dir, 'data.yaml'), 'w') as fw:
        fw.write(yaml.safe_dump(data))


def write_ymir_training_result(cfg: edict,
                               map50: float,
                               epoch: int,
                               weight_file: str) -> int:
    """
    cfg: ymir config
    results: (mp, mr, map50, map, loss)
    maps: map@0.5:0.95 for all classes
    epoch: stage
    weight_file: saved weight files, empty weight_file will save all files
    """
    model = cfg.param.model
    # use `rw.write_training_result` to save training result
    if weight_file:
        rw.write_model_stage(stage_name=f"{model}_{epoch}",
                             files=[osp.basename(weight_file)],
                             mAP=float(map50))
    else:
        # save other files with
        files = [osp.basename(f) for f in glob.glob(osp.join(cfg.ymir.output.models_dir, '*'))
                 if not f.endswith('.pt')] + ['last.pt', 'best.pt']

        rw.write_model_stage(stage_name=f"{model}_last_and_best",
                             files=files,
                             mAP=float(map50))
    return 0
