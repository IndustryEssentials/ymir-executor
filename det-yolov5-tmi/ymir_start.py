import argparse
import logging
import os.path as osp
import shutil
import subprocess
import sys
from typing import List

import cv2
from loguru import logger
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from utils.ymir_yolov5 import Ymir_Yolov5, convert_ymir_to_yolov5, get_weight_file, ymir_process_config


def get_args():
    parser = argparse.ArgumentParser('debug ...')
    parser.add_argument('--app', default=None,
                        help='training, mining or infer',
                        choices=['training', 'mining', 'infer'])
    parser.add_argument('--cfg', default=None,
                        help='set the /in/config.yaml')

    return parser.parse_args()


def start() -> int:
    env_config = env.get_current_env()

    logger.add(osp.join(env_config.output.root_dir, 'ymir_start.log'))

    logger.info(f'env_config: {env_config}')

    args = get_args()
    logger.info(f'args: {args}')

    if args.cfg is not None:
        default_cfg_file = env_config.input.config_file

        if osp.exists(default_cfg_file):
            shutil.copy(default_cfg_file, default_cfg_file + '.backup')

        shutil.copy(args.cfg, default_cfg_file)

    if args.app == 'training':
        _run_training(env_config)
    elif args.app == 'mining':
        _run_mining(env_config)
    elif args.app == 'infer':
        _run_infer(env_config)
    elif env_config.run_training:
        _run_training(env_config)
    elif env_config.run_mining:
        _run_mining(env_config)
    elif env_config.run_infer:
        _run_infer(env_config)

    return 0


def _run_training(env_config: env.EnvConfig) -> None:
    """
    sample function of training, which shows:
    1. how to get config file
    2. how to read training and validation datasets
    3. how to write logs
    4. how to write training result
    """
    # use `env.get_executor_config` to get config file for training
    executor_config = env.get_universal_config()
    # use `logging` or `print` to write log to console
    #   notice that logging.basicConfig is invoked at executor.env
    logging.info(f"training config: {executor_config}")

    logger.info('start convert ymir dataset to yolov5 dataset')
    out_dir = osp.join(env_config.output.root_dir, 'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    logger.info('convert ymir dataset to yolov5 dataset finished!!!')
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'])

    epochs = executor_config.get('epochs', 300)
    batch_size = executor_config.get('batch_size', 64)
    model = executor_config.get('model', 'yolov5s')
    img_size = executor_config.get('img_size', 640)
    weights = get_weight_file()

    models_dir = env_config.output.models_dir
    command = f'python train.py --epochs {epochs} ' + \
        f'--batch-size {batch_size} --data data.yaml --project /out ' + \
        f'--cfg models/{model}.yaml --name models --weights {weights} ' + \
        f'--img-size {img_size} --hyp data/hyps/hyp.scratch-low.yaml ' + \
        '--exist-ok'
    # use `monitor.write_monitor_logger` to write write task process percent to monitor.txt
    logger.info('start training ' + '*' * 50)
    logger.info(f'{command}')
    logger.info('*' * 80)

    # os.system(command)
    subprocess.check_output(command.split())
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'] + ymir_process_config['task'])

    # convert to onnx
    logging.info('convert to onnx ' + '*' * 50)
    opset = executor_config['opset']
    command = f'python export.py --weights {models_dir}/best.pt --opset {opset} --include onnx'
    logger.info(f'{command}')
    logger.info('*' * 80)
    # os.system(command)
    subprocess.check_output(command.split())
    # suppose we have a long time training, and have saved the final model

    shutil.copy(f'models/{model}.yaml', f'{models_dir}/{model}.yaml')

    # if task done, write 100% percent log
    logging.info('convert done')
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(env_config: env.EnvConfig) -> None:
    # use `env.get_executor_config` to get config file for training
    #   models are transfered in executor_config's model_params_path
    executor_config = env.get_executor_config()
    # use `logging` or `print` to write log to console
    logging.info(f"mining config: {executor_config}")

    logger.info('start convert ymir dataset to yolov5 dataset')
    out_dir = osp.join(env_config.output.root_dir, 'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    logger.info('convert ymir dataset to yolov5 dataset finished!!!')
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'])

    command = 'python mining/mining_cald.py'
    subprocess.check_output(command.split())

    # write mining result
    #   here we give a fake score to each assets
    # total_length = len(asset_paths)
    # mining_result = [(asset_path, index / total_length) for index, asset_path in enumerate(asset_paths)]
    # rw.write_mining_result(mining_result=mining_result)

    # if task done, write 100% percent log
    logging.info('mining done')
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(env_config: env.EnvConfig) -> None:
    # use `env.get_executor_config` to get config file for training
    #   models are transfered in executor_config's model_params_path
    executor_config = env.get_executor_config()
    # use `logging` or `print` to write log to console
    logging.info(f"infer config: {executor_config}")

    # generate data.yaml for infer
    logger.info('start convert ymir dataset to yolov5 dataset')
    out_dir = osp.join(env_config.output.root_dir, 'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    logger.info('convert ymir dataset to yolov5 dataset finished!!!')
    monitor.write_monitor_logger(percent=ymir_process_config['preprocess'])

    # use `monitor.write_monitor_logger` to write log to console and write task process percent to monitor.txt
    N = dr.items_count(env.DatasetType.CANDIDATE)
    logging.info(f"assets count: {N}")

    infer_result = dict()
    model = Ymir_Yolov5()
    idx = 0
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        img_path = osp.join(env_config.input.root_dir, env_config.input.assets_dir, asset_path)
        img = cv2.imread(img_path)

        result = model.infer(img)

        infer_result[asset_path] = result
        idx += 1
        monitor.write_monitor_logger(percent=ymir_process_config['preprocess'] + ymir_process_config['task'] * idx / N)

    # write infer result
    # fake_annotation = rw.Annotation(class_name=class_names[0], score=0.9, box=rw.Box(x=50, y=50, w=150, h=150))
    # infer_result = {asset_path: [fake_annotation] for asset_path in asset_paths}
    rw.write_infer_result(infer_result=infer_result)

    # if task done, write 100% percent log
    logging.info('infer done')
    monitor.write_monitor_logger(percent=1.0)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    sys.exit(start())
