import logging
import os
import subprocess
import sys

from easydict import EasyDict as edict

from mmdet.utils.util_ymir import get_merged_config
from ymir_exc import monitor


def start(cfg: edict) -> int:
    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training()
    elif cfg.ymir.run_mining or cfg.ymir.run_infer:
        if cfg.ymir.run_mining:
            _run_mining()
        if cfg.ymir.run_infer:
            _run_infer()
    else:
        logging.warning('no task running')

    return 0


def _run_training() -> None:
    command = 'python3 ymir_train.py'
    logging.info(f'start training: {command}')
    subprocess.run(command.split(), check=True)

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)
    logging.info("training finished")


def _run_mining() -> None:
    command = 'python3 ymir_mining.py'
    logging.info(f'start mining: {command}')
    subprocess.run(command.split(), check=True)
    logging.info("mining finished")

def _run_infer() -> None:
    command = 'python3 ymir_infer.py'
    logging.info(f'start infer: {command}')
    subprocess.run(command.split(), check=True)
    logging.info("infer finished")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    cfg = get_merged_config()
    os.environ.setdefault('YMIR_MODELS_DIR', cfg.ymir.output.models_dir)
    os.environ.setdefault('COCO_EVAL_TMP_FILE', os.path.join(
        cfg.ymir.output.root_dir, 'eval_tmp.json'))
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(start(cfg))
