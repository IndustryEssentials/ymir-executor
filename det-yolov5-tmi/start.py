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

from utils.ymir_yolov5 import (YmirStage, YmirYolov5, convert_ymir_to_yolov5,
                               download_weight_file, get_merged_config,
                               get_weight_file, get_ymir_process, write_ymir_training_result)


def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    else:
        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            mining_task_idx = 0
            infer_task_idx = 1
            task_num = 2
        else:
            mining_task_idx = 0
            infer_task_idx = 0
            task_num = 1

        if cfg.ymir.run_mining:
            _run_mining(cfg, mining_task_idx, task_num)
        if cfg.ymir.run_infer:
            _run_infer(cfg, infer_task_idx, task_num)

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
    save_period = max(1, min(epochs // 10, int(cfg.param.save_period)))
    args_options = cfg.param.args_options
    gpu_id = str(cfg.param.gpu_id)
    gpu_count = len(gpu_id.split(',')) if gpu_id else 0
    port = int(cfg.param.get('port', 29500))
    sync_bn = cfg.param.get('sync_bn', False)
    if isinstance(sync_bn, str):
        if sync_bn.lower() in ['f', 'false']:
            sync_bn = False
        elif sync_bn.lower() in ['t', 'true']:
            sync_bn = True
        else:
            raise Exception(f'unknown bool str sync_bn = {sync_bn}')

    weights = get_weight_file(cfg)
    if not weights:
        # download pretrained weight
        weights = download_weight_file(model)

    models_dir = cfg.ymir.output.models_dir

    commands = ['python3']
    if gpu_count == 0:
        device = 'cpu'
    elif gpu_count == 1:
        device = gpu_id
    else:
        device = gpu_id
        commands += f'-m torch.distributed.launch --nproc_per_node {gpu_count} --master_port {port}'.split()

    commands += ['train.py',
                 '--epochs', str(epochs),
                 '--batch-size', str(batch_size),
                 '--data', f'{out_dir}/data.yaml',
                 '--project', '/out',
                 '--cfg', f'models/{model}.yaml',
                 '--name', 'models', '--weights', weights,
                 '--img-size', str(img_size),
                 '--save-period', str(save_period),
                 '--device', device]

    if gpu_count > 1 and sync_bn:
        commands.append("--sync-bn")

    if args_options:
        commands += args_options.split()

    logging.info(f'start training: {commands}')

    subprocess.run(commands, check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=1.0))

    # 3. convert to onnx and save model weight to design directory
    opset = cfg.param.opset
    command = f'python3 export.py --weights {models_dir}/best.pt --opset {opset} --include onnx'
    logging.info(f'export onnx weight: {command}')
    subprocess.run(command.split(), check=True)

    # save hyperparameter
    shutil.copy(f'models/{model}.yaml', f'{models_dir}/{model}.yaml')
    write_ymir_training_result(cfg)
    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict, task_idx: int = 0, task_num: int = 1) -> None:
    # generate data.yaml for mining
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))

    command = 'python3 mining/mining_cald.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.POSTPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))


def _run_infer(cfg: edict, task_idx: int = 0, task_num: int = 1) -> None:
    # generate data.yaml for infer
    out_dir = cfg.ymir.output.root_dir
    convert_ymir_to_yolov5(cfg)
    logging.info(f'generate {out_dir}/data.yaml')
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))

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
            percent = get_ymir_process(stage=YmirStage.TASK, p=idx / N, task_idx=task_idx, task_num=task_num)
            monitor.write_monitor_logger(percent=percent)

    rw.write_infer_result(infer_result=infer_result)
    monitor.write_monitor_logger(percent=get_ymir_process(
        stage=YmirStage.PREPROCESS, p=1.0, task_idx=task_idx, task_num=task_num))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(start())
