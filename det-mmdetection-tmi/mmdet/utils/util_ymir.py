"""
utils function for ymir and yolov5
"""
import glob
import logging
import os
import os.path as osp
from enum import IntEnum
from typing import Any, List, Optional

import mmcv
import yaml
from easydict import EasyDict as edict
from mmcv import Config
from nptyping import NDArray, Shape, UInt8
from packaging.version import Version
from ymir_exc import env
from ymir_exc import result_writer as rw


class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2    # training/mining/infer
    POSTPROCESS = 3  # export model


BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_ymir_process(stage: YmirStage, p: float, task_idx: int = 0, task_num: int = 1) -> float:
    """
    stage: pre-process/task/post-process
    p: percent for stage
    task_idx: index for multiple tasks like mining (task_idx=0) and infer (task_idx=1)
    task_num: the total number of multiple tasks.
    """
    # const value for ymir process
    PREPROCESS_PERCENT = 0.1
    TASK_PERCENT = 0.8
    POSTPROCESS_PERCENT = 0.1

    if p < 0 or p > 1.0:
        raise Exception(f'p not in [0,1], p={p}')

    init = task_idx * 1.0 / task_num
    ratio = 1.0 / task_num
    if stage == YmirStage.PREPROCESS:
        return init + PREPROCESS_PERCENT * p * ratio
    elif stage == YmirStage.TASK:
        return init + (PREPROCESS_PERCENT + TASK_PERCENT * p) * ratio
    elif stage == YmirStage.POSTPROCESS:
        return init + (PREPROCESS_PERCENT + TASK_PERCENT + POSTPROCESS_PERCENT * p) * ratio
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


def _modify_mmdet_config(mmdet_cfg: Config, ymir_cfg: edict) -> Config:
    """
    useful for training process
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
                                filter_empty_gt=False
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
            if src_dataset_type in ['CocoDataset', 'YmirDataset']:
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
    if len(mmdet_cfg.log_config['hooks']) <= 1:
        mmdet_cfg.log_config['hooks'].append(tensorboard_logger)
    else:
        mmdet_cfg.log_config['hooks'][1].update(tensorboard_logger)

    # modify evaluation and interval
    interval = max(1, mmdet_cfg.runner.max_epochs // 30)
    mmdet_cfg.evaluation.interval = interval
    mmdet_cfg.evaluation.metric = ymir_cfg.param.get('metric', 'bbox')
    # TODO Whether to evaluating the AP for each class
    # mmdet_cfg.evaluation.classwise = True
    return mmdet_cfg


def get_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.pretrained_model_params or cfg.param.model_params_path
    """
    if cfg.ymir.run_training:
        model_params_path: List[str] = cfg.param.get('pretrained_model_params', [])
    else:
        model_params_path = cfg.param.get('model_params_path', [])

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith(('.pth', '.pt'))]

    # choose weight file by priority, best_xxx.pth > latest.pth > epoch_xxx.pth
    best_pth_files = [
        f for f in model_params_path if osp.basename(f).startswith('best_')]
    if len(best_pth_files) > 0:
        return max(best_pth_files, key=os.path.getctime)

    epoch_pth_files = [
        f for f in model_params_path if osp.basename(f).startswith(('epoch_', 'iter_'))]
    if len(epoch_pth_files) > 0:
        return max(epoch_pth_files, key=os.path.getctime)

    return ""


def write_ymir_training_result(last: bool = False, key_score: Optional[float] = None):
    YMIR_VERSION = os.environ.get('YMIR_VERSION', '1.2.0')
    if Version(YMIR_VERSION) >= Version('1.2.0'):
        _write_latest_ymir_training_result(last, key_score)
    else:
        _write_ancient_ymir_training_result(key_score)


def _write_latest_ymir_training_result(last: bool = False, key_score: Optional[float] = None):
    if key_score:
        logging.info(f'key_score is {key_score}')
    COCO_EVAL_TMP_FILE = os.getenv('COCO_EVAL_TMP_FILE')
    if COCO_EVAL_TMP_FILE is None:
        raise Exception(
            'please set valid environment variable COCO_EVAL_TMP_FILE to write result into json file')

    eval_result = mmcv.load(COCO_EVAL_TMP_FILE)
    # eval_result may be empty dict {}.
    map = eval_result.get('bbox_mAP_50', 0)

    WORK_DIR = os.getenv('YMIR_MODELS_DIR')
    if WORK_DIR is None or not osp.isdir(WORK_DIR):
        raise Exception(
            f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {WORK_DIR}')

    # assert only one model config file in work_dir
    result_files = [osp.basename(f) for f in glob.glob(
        osp.join(WORK_DIR, '*')) if osp.basename(f) != 'result.yaml']

    if last:
        # save all output file
        rw.write_model_stage(files=result_files,
                             mAP=float(map),
                             stage_name='last')
    else:
        # save newest weight file in format epoch_xxx.pth or iter_xxx.pth
        weight_files = [osp.join(WORK_DIR, f) for f in result_files if f.startswith(
            ('iter_', 'epoch_')) and f.endswith('.pth')]

        if len(weight_files) > 0:
            newest_weight_file = osp.basename(
                max(weight_files, key=os.path.getctime))

            stage_name = osp.splitext(newest_weight_file)[0]
            training_result_file = osp.join(WORK_DIR, 'result.yaml')
            if osp.exists(training_result_file):
                with open(training_result_file, 'r') as f:
                    training_result = yaml.safe_load(f)
                    model_stages = training_result.get('model_stages', {})
            else:
                model_stages = {}

            if stage_name not in model_stages:
                config_files = [f for f in result_files if f.endswith('.py')]
                rw.write_model_stage(files=[newest_weight_file] + config_files,
                                     mAP=float(map),
                                     stage_name=stage_name)


def _write_ancient_ymir_training_result(key_score: Optional[float] = None):
    if key_score:
        logging.info(f'key_score is {key_score}')

    COCO_EVAL_TMP_FILE = os.getenv('COCO_EVAL_TMP_FILE')
    if COCO_EVAL_TMP_FILE is None:
        raise Exception(
            'please set valid environment variable COCO_EVAL_TMP_FILE to write result into json file')

    eval_result = mmcv.load(COCO_EVAL_TMP_FILE)
    # eval_result may be empty dict {}.
    map = eval_result.get('bbox_mAP_50', 0)

    WORK_DIR = os.getenv('YMIR_MODELS_DIR')
    if WORK_DIR is None or not osp.isdir(WORK_DIR):
        raise Exception(
            f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {WORK_DIR}')

    # assert only one model config file in work_dir
    result_files = [osp.basename(f) for f in glob.glob(
        osp.join(WORK_DIR, '*')) if osp.basename(f) != 'result.yaml']

    training_result_file = osp.join(WORK_DIR, 'result.yaml')
    if osp.exists(training_result_file):
        with open(training_result_file, 'r') as f:
            training_result = yaml.safe_load(f)

        training_result['model'] = result_files
        training_result['map'] = max(map, training_result['map'])
    else:
        training_result = dict(model=result_files, map=map)

    with open(training_result_file, 'w') as f:
        yaml.safe_dump(training_result, f)
