"""
utils function for ymir and yolov5
"""
import glob
import os
import os.path as osp
import sys
from enum import IntEnum
from typing import Any, List, Tuple
from urllib.parse import urlparse

import mmcv
from mmcv import Config
from easydict import EasyDict as edict
from nptyping import NDArray, Shape, UInt8
from torch.hub import HASH_REGEX, _get_torch_home, download_url_to_file
from ymir_exc import env
from ymir_exc import result_writer as rw


class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2    # training/mining/infer
    POSTPROCESS = 3  # export model


BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_ymir_process(stage: YmirStage, p: float = 0.0) -> float:
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

def modify_mmdet_config(mmdet_cfg: Config, ymir_cfg: edict) -> Config:
    """
    - modify dataset config
    - modify model output channel
    """
    ### modify dataset config
    ymir_ann_files = dict(
        train=ymir_cfg.ymir.input.training_index_file,
        val=ymir_cfg.ymir.input.val_index_file,
        test=ymir_cfg.ymir.input.candidate_index_file
    )

    samples_per_gpu = ymir_cfg.param.samples_per_gpu
    workers_per_gpu = ymir_cfg.param.workers_per_gpu
    mmdet_cfg.data.samples_per_gpu = samples_per_gpu
    mmdet_cfg.data.workers_per_gpu = workers_per_gpu

    for split in ['train','val','test']:
        ymir_dataset_cfg=dict(type='YmirDataset',
            ann_file=ymir_ann_files[split],
            img_prefix=ymir_cfg.ymir.input.assets_dir,
            ann_prefix=ymir_cfg.ymir.input.annotations_dir,
            classes=ymir_cfg.param.class_names,
            data_root=ymir_cfg.ymir.input.root_dir,
            filter_empty_gt=False
            )
        ### modify dataset config
        mmdet_dataset_cfg = mmdet_cfg.data[split]
        if isinstance(mmdet_dataset_cfg, (list, tuple)):
            for x in mmdet_dataset_cfg:
                x.update(ymir_dataset_cfg)
        else:
            src_dataset_type = mmdet_dataset_cfg.type
            if src_dataset_type in ['CocoDataset']:
                mmdet_dataset_cfg.update(ymir_dataset_cfg)
            elif src_dataset_type in ['MultiImageMixDataset','RepeatDataset']:
                mmdet_dataset_cfg.dataset.update(ymir_dataset_cfg)
            else:
                raise Exception(f'unsupported source dataset type {src_dataset_type}')

    ### modify model output channel
    mmdet_model_cfg = mmdet_cfg.model.bbox_head
    mmdet_model_cfg.num_classes = len(ymir_cfg.param.class_names)

    ### epochs, checkpoint, tensorboard
    mmdet_cfg.runner.max_epochs = ymir_cfg.param.max_epochs
    mmdet_cfg.checkpoint_config['out_dir'] = ymir_cfg.ymir.output.models_dir
    tensorboard_logger = dict(type='TensorboardLoggerHook',
        log_dir = ymir_cfg.ymir.output.tensorboard_dir)
    mmdet_cfg.log_config['hooks'].append(tensorboard_logger)
    return mmdet_cfg

def get_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    """
    if cfg.ymir.run_training:
        model_params_path: List = cfg.param.pretrained_model_paths
    else:
        model_params_path: List = cfg.param.model_params_path

    model_dir = osp.join(cfg.ymir.input.root_dir,
                         cfg.ymir.input.models_dir)
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith('.pth')]

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    best_pth_files = [f for f in model_params_path if f.startswith('best_')]
    if len(best_pth_files) > 0:
        return get_newest_file(best_pth_files)

    epoch_pth_files = [f for f in model_params_path if f.startswith('epoch_')]
    if len(epoch_pth_files) > 0:
        return get_newest_file(epoch_pth_files)

    return ""


def download_weight_file(model: str) -> str:
    """
    download weight file from web if not exist.
    """
    model_to_url = dict(
        faster_rcnn_r50_fpn='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        faster_rcnn_r101_fpn='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth',
        yolox_tiny='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        yolox_s='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        yolox_l='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        yolox_x='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        yolox_nano='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    )

    url = model_to_url[model]
    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(
            url, cached_file))
        r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
        hash_prefix = r.group(1) if r else None
        download_url_to_file(
            url, cached_file, hash_prefix, progress=True)

    return cached_file


def update_training_result_file(key_score):
    COCO_EVAL_TMP_FILE = os.getenv('COCO_EVAL_TMP_FILE')
    if COCO_EVAL_TMP_FILE is None:
        raise Exception(
            'please set valid environment variable COCO_EVAL_TMP_FILE to write result into json file')

    results_per_category = mmcv.load(COCO_EVAL_TMP_FILE)

    work_dir = os.getenv('YMIR_MODELS_DIR')
    if work_dir is None or osp.isdir(work_dir):
        raise Exception(
            f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {work_dir}')

    # assert only one model config file in work_dir
    model_config_file = glob.glob(osp.join(work_dir, '*.py'))[0]
    weight_files = glob.glob(osp.join(work_dir, 'best_bbox_mAP_epoch_*.pth'))
    if len(weight_files) == 0:
        weight_files = glob.glob(osp.join(work_dir, 'epoch_*.pth'))

    if len(weight_files) == 0:
        raise Exception(f'no weight file found in {work_dir}')

    # sort the weight files by time, use the latest file.
    weight_files.sort(key=lambda fn: osp.getmtime(fn))
    model_weight_file = osp.basename(weight_files[-1])
    rw.write_training_result(model_names=[model_weight_file, osp.basename(model_config_file)],
                             mAP=key_score,
                             classAPs=results_per_category)
