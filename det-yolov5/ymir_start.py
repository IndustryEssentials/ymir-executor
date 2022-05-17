import logging
from xmlrpc.server import resolve_dotted_attribute
from loguru import logger 
import os
import os.path as osp
import sys
import time
from typing import List
import subprocess
import shutil

from executor import dataset_reader as dr, env, monitor, result_writer as rw
from utils.ymir_yolov5 import convert_ymir_to_yolov5

def start() -> int:
    env_config = env.get_current_env()

    logger.add(osp.join(env_config.output.root_dir,'ymir_start.log'))

    logger.info(f'env_config: {env_config}')
    if env_config.run_training:
        _run_training(env_config)
    # if env_config.run_mining:
    #     _run_mining(env_config)
    # if env_config.run_infer:
    #     _run_infer(env_config)

    return 0

def _run_training(env_config: env.EnvConfig) -> None:
    """
    sample function of training, which shows:
    1. how to get config file
    2. how to read training and validation datasets
    3. how to write logs
    4. how to write training result
    """
    #! use `env.get_executor_config` to get config file for training
    executor_config = env.get_executor_config()
    class_names: List[str] = executor_config['class_names']
    #! use `logging` or `print` to write log to console
    #   notice that logging.basicConfig is invoked at executor.env
    logging.info(f"training config: {executor_config}")
    
    logger.info(f'start convert ymir dataset to yolov5 dataset')
    monitor.write_monitor_logger(percent=0.01)
    out_dir = osp.join(env_config.output.root_dir,'yolov5_dataset')
    convert_ymir_to_yolov5(out_dir)
    logger.info(f'convert ymir dataset to yolov5 dataset finished!!!')
    monitor.write_monitor_logger(percent=0.1)

    
    epochs = executor_config['epochs']
    batch_size = executor_config['batch_size']
    model = executor_config['model']
    img_size = executor_config['img_size']
    models_dir = env_config.output.models_dir
    command = f'python train.py --epochs {epochs} ' + \
        f'--batch-size {batch_size} --data data.yaml --project /out ' + \
        f'--cfg models/{model}.yaml --name models ' + \
        f'--img-size {img_size} --hyp data/hyps/hyp.scratch-low.yaml ' + \
        '--exist-ok'
    #! use `monitor.write_monitor_logger` to write write task process percent to monitor.txt
    logger.info(f'start training '+'*'*50)
    logger.info(f'{command}')
    logger.info('*'*80)

    # os.system(command)
    subprocess.check_output(command.split())
    monitor.write_monitor_logger(percent=0.9)

    # convert to onnx 
    logging.info('convert to onnx '+'*'*50)
    opset = executor_config['opset']
    command = f'python export.py --weights {models_dir}/best.pt --opset {opset} --include onnx'
    logger.info(f'{command}')
    logger.info('*'*80)
    # os.system(command)
    subprocess.check_output(command.split())
    # suppose we have a long time training, and have saved the final model

    shutil.copy(f'models/{model}.yaml', f'{models_dir}/{model}.yaml')

    #! if task done, write 100% percent log
    logging.info('convert done')
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(env_config: env.EnvConfig) -> None:
    #! use `env.get_executor_config` to get config file for training
    #   models are transfered in executor_config's model_params_path
    executor_config = env.get_executor_config()
    idle_seconds: float = executor_config.get('idle_seconds', 60)
    trigger_crash: bool = executor_config.get('trigger_crash', False)
    #! use `logging` or `print` to write log to console
    logging.info(f"mining config: {executor_config}")

    #! use `dataset_reader.item_paths` to read candidate dataset items
    #   note that annotations path will be empty str if there's no annotations in that dataset
    asset_paths = []
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        logging.info(f"asset: {asset_path}")
        asset_paths.append(asset_path)

    if len(asset_paths) == 0:
        raise ValueError('empty asset paths')

    #! use `monitor.write_monitor_logger` to write task process to monitor.txt
    logging.info(f"assets count: {len(asset_paths)}")
    monitor.write_monitor_logger(percent=0.5)

    _dummy_work(idle_seconds=idle_seconds, trigger_crash=trigger_crash)

    #! write mining result
    #   here we give a fake score to each assets
    total_length = len(asset_paths)
    mining_result = [(asset_path, index / total_length) for index, asset_path in enumerate(asset_paths)]
    rw.write_mining_result(mining_result=mining_result)

    #! if task done, write 100% percent log
    logging.info('mining done')
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(env_config: env.EnvConfig) -> None:
    #! use `env.get_executor_config` to get config file for training
    #   models are transfered in executor_config's model_params_path
    executor_config = env.get_executor_config()
    class_names = executor_config['class_names']
    idle_seconds: float = executor_config.get('idle_seconds', 60)
    trigger_crash: bool = executor_config.get('trigger_crash', False)
    #! use `logging` or `print` to write log to console
    logging.info(f"infer config: {executor_config}")

    #! use `dataset_reader.item_paths` to read candidate dataset items
    #   note that annotations path will be empty str if there's no annotations in that dataset
    asset_paths: List[str] = []
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        logging.info(f"asset: {asset_path}")
        asset_paths.append(asset_path)

    if len(asset_paths) == 0 or len(class_names) == 0:
        raise ValueError('empty asset paths or class names')

    #! use `monitor.write_monitor_logger` to write log to console and write task process percent to monitor.txt
    logging.info(f"assets count: {len(asset_paths)}")
    monitor.write_monitor_logger(percent=0.5)

    _dummy_work(idle_seconds=idle_seconds, trigger_crash=trigger_crash)

    #! write infer result
    fake_annotation = rw.Annotation(class_name=class_names[0], score=0.9, box=rw.Box(x=50, y=50, w=150, h=150))
    infer_result = {asset_path: [fake_annotation] for asset_path in asset_paths}
    rw.write_infer_result(infer_result=infer_result)

    #! if task done, write 100% percent log
    logging.info('infer done')
    monitor.write_monitor_logger(percent=1.0)


def _dummy_work(idle_seconds: float, trigger_crash: bool = False, gpu_memory_size: int = 0) -> None:
    if idle_seconds > 0:
        time.sleep(idle_seconds)
    if trigger_crash:
        raise RuntimeError('app crashed')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)
    sys.exit(start())
