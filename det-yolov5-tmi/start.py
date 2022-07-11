import logging
import os
import os.path as osp
import shutil
import subprocess
import sys

import cv2
from easydict import EasyDict as edict
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from utils.ymir_yolov5 import (YmirStage, YmirYolov5, convert_ymir_to_yolov5, download_weight_file, get_merged_config,
                               get_weight_file, get_ymir_process)


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    else:
        if cfg.ymir.run_mining:
            _run_mining(cfg)
        if cfg.ymir.run_infer:
            _run_infer(cfg)

    return 0


def _run_training(cfg: edict) -> None:
    """
    function for training task
    1. convert dataset
    2. training model
    3. save model weight/hyperparameter/... to design directory
    """
    # 1. convert dataset
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    # 2. training model
    epochs = cfg.param.epochs
    batch_size = cfg.param.batch_size
    model = cfg.param.model
    img_size = cfg.param.img_size
    save_period = cfg.param.save_period
    args_options = cfg.param.args_options
    gpu_id = str(cfg.param.gpu_id)
    gpu_count = len(gpu_id.split(',')) if gpu_id else 0
    port = int(cfg.param.port)
    sync_bn = cfg.param.sync_bn
    weights = get_weight_file(cfg)
    if not weights:
        # download pretrained weight
        weights = download_weight_file(model)

    models_dir = cfg.ymir.output.models_dir

    if gpu_count == 0:
        command = f'python3 train.py --epochs {epochs} ' + \
            f'--batch-size {batch_size} --data {out_dir}/data.yaml --project /out ' + \
            f'--cfg models/{model}.yaml --name models --weights {weights} ' + \
            f'--img-size {img_size} ' + \
            f'--save-period {save_period} ' + \
            f'--devices cpu'
    elif gpu_count == 1:
        command = f'python3 train.py --epochs {epochs} ' + \
            f'--batch-size {batch_size} --data {out_dir}/data.yaml --project /out ' + \
            f'--cfg models/{model}.yaml --name models --weights {weights} ' + \
            f'--img-size {img_size} ' + \
            f'--save-period {save_period} ' + \
            f'--devices {gpu_id}'
    else:
        command = f'python3 -m torch.distributed.launch --nproc_per_node {gpu_count} ' + \
            f'--master_port {port} train.py --epochs {epochs} ' + \
            f'--batch-size {batch_size} --data {out_dir}/data.yaml --project /out ' + \
            f'--cfg models/{model}.yaml --name models --weights {weights} ' + \
            f'--img-size {img_size} ' + \
            f'--save-period {save_period} ' + \
            f'--devices {gpu_id}'

        if sync_bn:
            command += " --sync-bn"

    if args_options:
        command += f" {args_options}"

    logging.info(f'start training: {command}')

    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=1.0))

    # 3. convert to onnx and save model weight to design directory
    opset = cfg.param.opset
    command = f'python3 export.py --weights {models_dir}/best.pt --opset {opset} --include onnx'
    logging.info(f'export onnx weight: {command}')
    subprocess.run(command.split(), check=True)

    # save hyperparameter
    shutil.copy(f'models/{model}.yaml', f'{models_dir}/{model}.yaml')

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict()) -> None:
    # generate data.yaml for mining
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    command = 'python3 mining/mining_cald.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(cfg: edict) -> None:
    # generate data.yaml for infer
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    N = dr.items_count(env.DatasetType.CANDIDATE)
    infer_result = dict()
    model = YmirYolov5(cfg)
    idx = -1

    monitor_gap = max(1, N // 100)
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        img = cv2.imread(asset_path)
        result = model.infer(img)
        infer_result[asset_path] = result
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
