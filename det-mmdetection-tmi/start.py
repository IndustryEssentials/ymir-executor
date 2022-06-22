import glob
import logging
import os
import subprocess
import sys

import cv2
import yaml
from easydict import EasyDict as edict
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from mmdet.utils.util_ymir import (YmirStage, get_merged_config,
                                   get_ymir_process)
from ymir_infer import YmirModel, mmdet_result_to_ymir


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    elif cfg.ymir.run_mining:
        _run_mining(cfg)
    elif cfg.ymir.run_infer:
        _run_infer(cfg)
    else:
        logging.warning('no task running')

    return 0


def _run_training(cfg: edict) -> None:
    """
    function for training task
    1. convert dataset
    2. training model
    3. save model weight/hyperparameter/... to design directory
    """
    command = 'python3 ymir_train.py'
    logging.info(f'start training: {command}')
    subprocess.run(command.split(), check=True)

    work_dir = cfg.ymir.output.models_dir
    result_files = glob.glob(os.path.join(work_dir, '*'))

    training_result_file = cfg.ymir.output.training_result_file
    with open(training_result_file, 'r') as fp:
        best_result = yaml.safe_load(fp)

    # save the last checkpoint
    rw.write_training_result(model_names=[os.path.basename(f) for f in result_files],
                             mAP=best_result['map'],
                             classAPs=best_result['class_aps'])

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict()) -> None:
    command = 'python3 mining/mining_cald.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(cfg: edict) -> None:
    N = dr.items_count(env.DatasetType.CANDIDATE)
    infer_result = dict()
    model = YmirModel(cfg)
    idx = -1

    # write infer result
    monitor_gap = max(1, N // 100)
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        img = cv2.imread(asset_path)
        result = model.infer(img)
        infer_result[asset_path] = mmdet_result_to_ymir(result, cfg.param.class_names)
        idx += 1

        if idx % monitor_gap == 0:
            percent = get_ymir_process(stage=YmirStage.TASK, p=idx / N)
            monitor.write_monitor_logger(percent=percent)

    rw.write_infer_result(infer_result=infer_result)
    monitor.write_monitor_logger(percent=1.0)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(start())