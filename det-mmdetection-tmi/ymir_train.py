import glob
import logging
import os
import os.path as osp
import subprocess
import sys

from easydict import EasyDict as edict
from ymir_exc import monitor
from mmdet.utils.util_ymir import get_merged_config, get_weight_file, download_weight_file, get_ymir_process, YmirStage, update_training_result_file


def main(cfg: edict) -> int:
    ### default ymir config
    gpu_id = cfg.param.get("gpu_id", '0')
    num_gpus = len(gpu_id.split(","))
    if num_gpus == 0:
        raise Exception(f'gpu_id = {gpu_id} is not valid, eg: 0 or 2,4')

    classes = cfg.param.class_names
    num_classes = len(classes)
    if num_classes==0:
        raise Exception('not find class_names in config!')

    # weight_file = get_weight_file(cfg)
    # if not weight_file:
    #     weight_file = download_weight_file(model)

    ### user define config
    # learning_rate = cfg.param.learning_rate
    # epochs = cfg.param.max_epochs

    # samples_per_gpu = cfg.param.samples_per_gpu
    # workers_per_gpu = min(4, max(1, samples_per_gpu//2))

    ### mmcv args config
    config_file = cfg.param.get("config_file")
    args_options = cfg.param.get("args",None)
    cfg_options = cfg.param.get("cfg_options",None)

    monitor.write_monitor_logger(percent=get_ymir_process(YmirStage.PREPROCESS, p=0.2))

    work_dir = cfg.ymir.output.models_dir
    if num_gpus == 0:
        # view https://mmdetection.readthedocs.io/en/stable/1_exist_data_model.html#training-on-cpu
        os.environ.setdefault('CUDA_VISIBLE_DEVICES',"-1")
        cmd = f"python tools/train.py {config_file} " + \
            f"--work-dir {work_dir}"
    elif num_gpus == 1:
        cmd = f"python tools/train.py {config_file} " + \
            f"--work-dir {work_dir} --gpu-id {gpu_id}"
    else:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_id)
        port = cfg.param.get('port')
        os.environ.setdefault('PORT', port)
        cmd = f"./tools/dist_train.sh {config_file} {num_gpus} " + \
            f"--work-dir {work_dir}"

    if args_options:
        cmd +=f" {args_options}"

    if cfg_options:
        cmd +=f" --cfg-options {cfg_options}"

    logging.info(f"training command: {cmd}")
    subprocess.run(cmd.split(), check=True)

    # eval_hooks will generate training_result_file if current map is best.
    # create a fake map = 0 if no training_result_file generate in eval_hooks
    if not osp.exists(cfg.ymir.output.training_result_file):
        update_training_result_file(0)

    return 0

if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('YMIR_MODELS_DIR',cfg.ymir.output.models_dir)
    os.environ.setdefault('COCO_EVAL_TMP_FILE', osp.join(cfg.ymir.output.root_dir,'eval_tmp.json'))
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(main(cfg))
