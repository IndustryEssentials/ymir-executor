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
    # default ymir config
    gpu_id = cfg.param.get("gpu_id", '0')
    num_gpus = len(gpu_id.split(","))
    if num_gpus == 0:
        raise Exception(f'gpu_id = {gpu_id} is not valid, eg: 0 or 2,4')

    classes = cfg.param.class_names
    num_classes = len(classes)
    model = cfg.param.model
    if num_classes==0:
        raise Exception('not find class_names in config!')

    weight_file = get_weight_file(cfg)
    if not weight_file:
        weight_file = download_weight_file(model)

    # user define config
    learning_rate = cfg.param.learning_rate
    epochs = cfg.param.epochs

    samples_per_gpu = cfg.param.samples_per_gpu
    workers_per_gpu = min(4, max(1, samples_per_gpu//2))

    supported_models = []
    if model.startswith("faster_rcnn"):
        files = glob.glob(
            osp.join('configs/faster_rcnn/faster_rcnn_*_ymir_coco.py'))
        supported_models = ['faster_rcnn_r50_fpn', 'faster_rcnn_r101_fpn']
    elif model.startswith("yolox"):
        files = glob.glob(osp.join('configs/yolox/yolox_*_8x8_300e_ymir_coco.py'))
        supported_models = ['yolox_nano', 'yolox_tiny',
                            'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x']
    else:
        files = glob.glob(osp.join('configs/*/*_ymir_coco.py'))
        supported_models = [osp.basename(f) for f in files]

    assert model in supported_models, f'unknown model {model}, not in {supported_models}'

    # modify base config file
    base_config_file = './configs/_base_/datasets/ymir_coco.py'

    modify_dict = dict(
        classes=classes,
        num_classes=num_classes,
        max_epochs=epochs,
        lr=learning_rate,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        data_root=cfg.ymir.input.root_dir,
        img_prefix=cfg.ymir.input.assets_dir,
        ann_prefix=cfg.ymir.input.annotations_dir,
        train_ann_file=cfg.ymir.input.training_index_file,
        val_ann_file=cfg.ymir.input.val_index_file,
        tensorboard_dir=cfg.ymir.output.tensorboard_dir,
        work_dir=cfg.ymir.output.models_dir,
        checkpoints_path=weight_file
    )

    logging.info(f'modified config is {modify_dict}')
    with open(base_config_file, 'r') as fp:
        lines = fp.readlines()

    fw = open(base_config_file, 'w')
    for line in lines:
        for key in modify_dict:
            if line.startswith((f"{key}=", f"{key} =")):
                value = modify_dict[key]
                if isinstance(value, str):
                    line = f"{key} = '{value}' \n"
                else:
                    line = f"{key} = {value} \n"
                break
        fw.write(line)
    fw.close()

    # train_config_file will use the config in base_config_file
    train_config_file = ''
    for f in files:
        if osp.basename(f).startswith(model):
            train_config_file = f

    monitor.write_monitor_logger(percent=get_ymir_process(YmirStage.PREPROCESS, p=0.2))

    work_dir = cfg.ymir.output.models_dir
    if num_gpus == 1:
        cmd = f"python tools/train.py {train_config_file} " + \
            f"--work-dir {work_dir} --gpu-id {gpu_id}"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        cmd = f"./tools/dist_train.sh {train_config_file} {num_gpus} " + \
            f"--work-dir {work_dir}"

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
    os.environ.setdefault('COCO_EVAL_TMP_FILE', '')
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(main(cfg))
