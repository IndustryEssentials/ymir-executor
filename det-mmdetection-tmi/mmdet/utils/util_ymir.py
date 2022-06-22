"""
utils function for ymir and yolov5
"""
import glob
import os
import os.path as osp
from enum import IntEnum
from typing import Any, List

import mmcv
from easydict import EasyDict as edict
from mmcv import Config
from nptyping import NDArray, Shape, UInt8
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
    - modify epochs, checkpoint, tensorboard config
    """
    # modify dataset config
    ymir_ann_files = dict(
        train=ymir_cfg.ymir.input.training_index_file,
        val=ymir_cfg.ymir.input.val_index_file,
        test=ymir_cfg.ymir.input.candidate_index_file
    )

    # validation may augment the image and use more gpu
    # so set smaller samples_per_gpu for validation
    samples_per_gpu = ymir_cfg.param.samples_per_gpu
    workers_per_gpu = ymir_cfg.param.workers_per_gpu
    mmdet_cfg.data.samples_per_gpu = samples_per_gpu
    mmdet_cfg.data.workers_per_gpu = workers_per_gpu

    for split in ['train', 'val', 'test']:
        ymir_dataset_cfg = dict(type='YmirDataset',
                                ann_file=ymir_ann_files[split],
                                img_prefix=ymir_cfg.ymir.input.assets_dir,
                                ann_prefix=ymir_cfg.ymir.input.annotations_dir,
                                classes=ymir_cfg.param.class_names,
                                data_root=ymir_cfg.ymir.input.root_dir,
                                filter_empty_gt=False,
                                samples_per_gpu=samples_per_gpu if split == 'train' else max(
                                    1, samples_per_gpu//2),
                                workers_per_gpu=workers_per_gpu if split == 'train' else max(
                                    1, workers_per_gpu//2)
                                )
        # modify dataset config for `split`
        mmdet_dataset_cfg = mmdet_cfg.data.get(split, None)
        if mmdet_dataset_cfg is None:
            continue

        if isinstance(mmdet_dataset_cfg, (list, tuple)):
            for x in mmdet_dataset_cfg:
                x.update(ymir_dataset_cfg)
        else:
            src_dataset_type = mmdet_dataset_cfg.type
            if src_dataset_type in ['CocoDataset']:
                mmdet_dataset_cfg.update(ymir_dataset_cfg)
            elif src_dataset_type in ['MultiImageMixDataset', 'RepeatDataset']:
                mmdet_dataset_cfg.dataset.update(ymir_dataset_cfg)
            else:
                raise Exception(
                    f'unsupported source dataset type {src_dataset_type}')

    # modify model output channel
    mmdet_model_cfg = mmdet_cfg.model.bbox_head
    mmdet_model_cfg.num_classes = len(ymir_cfg.param.class_names)

    # modify epochs, checkpoint, tensorboard config
    if ymir_cfg.param.get('max_epochs', None):
        mmdet_cfg.runner.max_epochs = ymir_cfg.param.max_epochs
    mmdet_cfg.checkpoint_config['out_dir'] = ymir_cfg.ymir.output.models_dir
    tensorboard_logger = dict(type='TensorboardLoggerHook',
                              log_dir=ymir_cfg.ymir.output.tensorboard_dir)
    mmdet_cfg.log_config['hooks'].append(tensorboard_logger)

    # modify evaluation and interval
    interval = max(1, mmdet_cfg.runner.max_epoch//30)
    mmdet_cfg.evaluation.interval = interval
    # Whether to evaluating the AP for each class
    mmdet_cfg.evaluation.classwise = True
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

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith(('.pth', '.pt'))]

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    best_pth_files = [
        f for f in model_params_path if osp.basename(f).startswith('best_')]
    if len(best_pth_files) > 0:
        return max(best_pth_files, key=os.path.getctime)

    epoch_pth_files = [
        f for f in model_params_path if osp.basename(f).startswith('epoch_')]
    if len(epoch_pth_files) > 0:
        return max(epoch_pth_files, key=os.path.getctime)

    return ""


def update_training_result_file(key_score):
    COCO_EVAL_TMP_FILE = os.getenv('COCO_EVAL_TMP_FILE')
    if COCO_EVAL_TMP_FILE is None:
        raise Exception(
            'please set valid environment variable COCO_EVAL_TMP_FILE to write result into json file')

    results_per_category = mmcv.load(COCO_EVAL_TMP_FILE)

    work_dir = os.getenv('YMIR_MODELS_DIR')
    if work_dir is None or not osp.isdir(work_dir):
        raise Exception(
            f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {work_dir}')

    # assert only one model config file in work_dir
    result_files = glob.glob(osp.join(work_dir, '*'))
    rw.write_training_result(model_names=[osp.basename(f) for f in result_files],
                             mAP=key_score,
                             classAPs=results_per_category)
