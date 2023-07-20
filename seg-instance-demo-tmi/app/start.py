import logging
import os
import random
import sys
import time
from typing import List

import cv2
import numpy as np
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
from ymir_exc import monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import get_merged_config

from result_to_coco import convert


def start() -> int:
    cfg = get_merged_config()

    if cfg.ymir.run_training:
        _run_training(cfg)
    if cfg.ymir.run_mining:
        _run_mining(cfg)
    if cfg.ymir.run_infer:
        _run_infer(cfg)

    return 0


def _run_training(cfg: edict) -> None:
    """sample function of training

    which shows:
    - how to get config file
    - how to read training and validation datasets
    - how to write logs
    - how to write training result
    """
    # use `env.get_executor_config` to get config file for training
    gpu_id: str = cfg.param.get('gpu_id')
    class_names: List[str] = cfg.param.get('class_names')
    expected_maskap: float = cfg.param.get('expected_maskap', 0.6)
    idle_seconds: float = cfg.param.get('idle_seconds', 60)
    trigger_crash: bool = cfg.param.get('trigger_crash', False)
    # use `logging` or `print` to write log to console
    #   notice that logging.basicConfig is invoked at executor.env
    logging.info(f'gpu device: {gpu_id}')
    logging.info(f'dataset class names: {class_names}')
    logging.info(f"training config: {cfg.param}")

    # count for image and annotation file
    with open(cfg.ymir.input.training_index_file, 'r') as fp:
        lines = fp.readlines()

    valid_image_count = 0
    valid_ann_count = 0

    N = len(lines)
    monitor_gap = max(1, N // 100)
    for idx, line in enumerate(lines):
        asset_path, annotation_path = line.strip().split()
        if os.path.isfile(asset_path):
            valid_image_count += 1

        if os.path.isfile(annotation_path):
            valid_ann_count += 1

        # use `monitor.write_monitor_logger` to write write task process percent to monitor.txt
        if idx % monitor_gap == 0:
            monitor.write_monitor_logger(percent=0.2 * idx / N)

    logging.info(f'total image-ann pair: {N}')
    logging.info(f'valid images: {valid_image_count}')
    logging.info(f'valid annotations: {valid_ann_count}')

    # use `monitor.write_monitor_logger` to write write task process percent to monitor.txt
    monitor.write_monitor_logger(percent=0.2)

    # suppose we have a long time training, and have saved the final model
    # model output dir: os.path.join(cfg.ymir.output.models_dir, your_stage_name)
    stage_dir = os.path.join(cfg.ymir.output.models_dir, 'epoch10')
    os.makedirs(stage_dir, exist_ok=True)
    with open(os.path.join(stage_dir, 'epoch10.pt'), 'w') as f:
        f.write('fake model weight')
    with open(os.path.join(stage_dir, 'config.py'), 'w') as f:
        f.write('fake model config file')
    # use `rw.write_model_stage` to save training result
    rw.write_model_stage(stage_name='epoch10',
                         files=['epoch10.pt', 'config.py'],
                         evaluation_result=dict(maskAP=random.random() / 2))

    _dummy_work(idle_seconds=idle_seconds, trigger_crash=trigger_crash)

    write_tensorboard_log(cfg.ymir.output.tensorboard_dir)

    stage_dir = os.path.join(cfg.ymir.output.models_dir, 'epoch20')
    os.makedirs(stage_dir, exist_ok=True)
    with open(os.path.join(stage_dir, 'epoch20.pt'), 'w') as f:
        f.write('fake model weight')
    with open(os.path.join(stage_dir, 'config.py'), 'w') as f:
        f.write('fake model config file')
    rw.write_model_stage(stage_name='epoch20',
                         files=['epoch20.pt', 'config.py'],
                         evaluation_result=dict(maskAP=expected_maskap))

    # if task done, write 100% percent log
    logging.info('training done')
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict) -> None:
    # use `cfg.param` to get config file for training
    #  pretrained models in `cfg.ymir.input.models_dir`
    gpu_id: str = cfg.param.get('gpu_id')
    class_names: List[str] = cfg.param.get('class_names')
    idle_seconds: float = cfg.param.get('idle_seconds', 60)
    trigger_crash: bool = cfg.param.get('trigger_crash', False)
    # use `logging` or `print` to write log to console
    logging.info(f"mining config: {cfg.param}")
    logging.info(f'gpu device: {gpu_id}')
    logging.info(f'dataset class names: {class_names}')

    # use `cfg.input.candidate_index_file` to read candidate dataset items
    #   note that annotations path will be empty str if there's no annotations in that dataset
    # count for image files
    with open(cfg.ymir.input.candidate_index_file, 'r') as fp:
        lines = fp.readlines()

    valid_images = []
    valid_image_count = 0
    for line in lines:
        if os.path.isfile(line.strip()):
            valid_image_count += 1
            valid_images.append(line.strip())

    # use `monitor.write_monitor_logger` to write task process to monitor.txt
    logging.info(f"assets count: {len(lines)}, valid: {valid_image_count}")
    monitor.write_monitor_logger(percent=0.2)

    _dummy_work(idle_seconds=idle_seconds, trigger_crash=trigger_crash)

    # write mining result
    #   here we give a fake score to each assets
    total_length = len(valid_images)
    mining_result = []
    for index, asset_path in enumerate(valid_images):
        mining_result.append((asset_path, index / total_length))
        time.sleep(0.1)
        monitor.write_monitor_logger(percent=0.2 + 0.8 * index / valid_image_count)

    rw.write_mining_result(mining_result=mining_result)

    # if task done, write 100% percent log
    logging.info('mining done')
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(cfg: edict) -> None:
    # use `cfg.param` to get config file for training
    #   models are transfered in `cfg.ymir.input.models_dir` model_params_path
    class_names = cfg.param.get('class_names')
    idle_seconds: float = cfg.param.get('idle_seconds', 60)
    trigger_crash: bool = cfg.param.get('trigger_crash', False)
    seed: int = cfg.param.get('seed', 15)
    # use `logging` or `print` to write log to console
    logging.info(f"infer config: {cfg.param}")

    # use `cfg.ymir.input.candidate_index_file` to read candidate dataset items
    #   note that annotations path will be empty str if there's no annotations in that dataset
    with open(cfg.ymir.input.candidate_index_file, 'r') as fp:
        lines = fp.readlines()

    valid_images = []
    invalid_images = []
    valid_image_count = 0
    for line in lines:
        if os.path.isfile(line.strip()):
            valid_image_count += 1
            valid_images.append(line.strip())
        else:
            invalid_images.append(line.strip())

    # use `monitor.write_monitor_logger` to write log to console and write task process percent to monitor.txt
    logging.info(f"assets count: {len(lines)}, valid: {valid_image_count}")
    monitor.write_monitor_logger(percent=0.2)

    _dummy_work(idle_seconds=idle_seconds, trigger_crash=trigger_crash)

    # write infer result
    random.seed(seed)
    results = []

    fake_mask_num = min(len(class_names), 10)
    for iter, img_file in enumerate(valid_images):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(shape=img.shape[0:2], dtype=np.uint8)
        for idx in range(fake_mask_num):
            percent = 100 * idx / fake_mask_num
            value = np.percentile(img, percent)
            mask[img > value] = idx + 1

        results.append(dict(image=img_file, result=mask))

        # real-time monitor
        monitor.write_monitor_logger(percent=0.2 + 0.8 * iter / valid_image_count)

    coco_results = convert(cfg, results, True)
    rw.write_infer_result(infer_result=coco_results, algorithm='segmentation')

    # if task done, write 100% percent log
    logging.info('infer done')
    monitor.write_monitor_logger(percent=1.0)


def _dummy_work(idle_seconds: float, trigger_crash: bool = False) -> None:
    if idle_seconds > 0:
        time.sleep(idle_seconds)
    if trigger_crash:
        raise RuntimeError('app crashed')


def write_tensorboard_log(tensorboard_dir: str) -> None:
    tb_log = SummaryWriter(tensorboard_dir)

    total_epoch = 30
    for e in range(total_epoch):
        tb_log.add_scalar("fake_loss", 10 / (1 + e), e)
        time.sleep(1)
        monitor.write_monitor_logger(percent=e / total_epoch)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    sys.exit(start())
