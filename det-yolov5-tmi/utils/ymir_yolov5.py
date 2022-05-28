"""
convert ymir dataset to yolov5 dataset
"""
import os
import os.path as osp
import shutil

import imagesize
import numpy as np
import torch
import yaml
from loguru import logger
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.datasets import img2label_paths
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

# const value for process
ymir_process_config = dict(
    preprocess=0.1,
    task=0.8,
    postprocess=0.1)


def get_weight_file() -> str:
    executor_config = env.get_universal_config()
    path_config = env.get_current_env()

    weights = None
    model_params_path = executor_config['model_params_path']
    model_dir = osp.join(path_config.input.root_dir,
                         path_config.input.models_dir)
    model_params_path = [p for p in model_params_path if osp.exists(osp.join(model_dir, p))]
    if 'best.pt' in model_params_path:
        weights = osp.join(model_dir, 'best.pt')
    else:
        for f in model_params_path:
            if f.endswith('.pt'):
                weights = osp.join(model_dir, f)
                break

    if weights is None:
        model_name = env.get_universal_config()['model']
        weights = f'{model_name}.pt'
        logger.info(f'cannot find pytorch weight in {model_params_path}, use {weights} instead')

    return weights


class Ymir_Yolov5():
    def __init__(self):
        executor_config = env.get_executor_config()
        gpu_id = executor_config.get('gpu_id', '0')
        gpu_num = len(gpu_id.split(','))
        if gpu_num == 0:
            device = 'cpu'
        else:
            device = gpu_id
        device = select_device(device)

        self.model = self.init_detector(device)
        self.device = device
        self.class_names = executor_config['class_names']

        self.stride = self.model.stride
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz, s=self.stride)
        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup

        self.img_size = imgsz

    def init_detector(self, device):
        weights = get_weight_file()

        if not osp.exists(weights):
            logger.info(f'try to download {weights} ' + '.' * 30)

        model = DetectMultiBackend(weights=weights,
                                   device=device,
                                   dnn=False,  # not use opencv dnn for onnx inference
                                   data='data.yaml')  # dataset.yaml path

        return model

    def predict(self, img):
        # preprocess
        # img0 = cv2.imread(path)  # BGR
        # Padded resize
        img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

        # Convert
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)

        img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
        if len(img1.shape) == 3:
            img1 = img1[None]  # expand for batch dim

        pred = self.model(img1)

        # postprocess
        conf_thres = 0.25
        iou_thres = 0.45
        classes = None
        agnostic_nms = False
        max_det = 1000

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        result = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()
                result.append(det)

        # xyxy, conf, cls
        if len(result) > 0:
            result = torch.cat(result, dim=0)
            result = result.data.cpu().numpy()

        return result

    def infer(self, img):
        anns = []
        result = self.predict(img)
        if result == []:
            anns = []
        else:
            for i in range(result.shape[0]):
                xmin, ymin, xmax, ymax, conf, cls = result[i, :6].tolist()
                ann = rw.Annotation(class_name=self.class_names[int(cls)], score=conf, box=rw.Box(
                    x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))

                anns.append(ann)

        return anns


def digit(x):
    if x < 10:
        return 1
    else:
        return 1 + digit(x // 10)


def convert_ymir_to_yolov5(root_dir, args=None):
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(osp.join(root_dir, 'images'), exist_ok=True)
    os.makedirs(osp.join(root_dir, 'labels'), exist_ok=True)

    if args is None:
        env_config = env.get_current_env()
        if env_config.run_training:
            train_data_size = dr.items_count(env.DatasetType.TRAINING)
            val_data_size = dr.items_count(env.DatasetType.VALIDATION)
            N = len(str(train_data_size + val_data_size))
            splits = ['train', 'val']
        elif env_config.run_mining:
            N = dr.items_count(env.DatasetType.CANDIDATE)
            splits = ['test']
        elif env_config.run_infer:
            N = dr.items_count(env.DatasetType.CANDIDATE)
            splits = ['test']
    else:
        if args.app == 'training':
            train_data_size = dr.items_count(env.DatasetType.TRAINING)
            val_data_size = dr.items_count(env.DatasetType.VALIDATION)
            N = len(str(train_data_size + val_data_size))
            splits = ['train', 'val']
        else:
            N = dr.items_count(env.DatasetType.CANDIDATE)
            splits = ['test']

    idx = 0
    DatasetTypeDict = dict(train=env.DatasetType.TRAINING,
                           val=env.DatasetType.VALIDATION,
                           test=env.DatasetType.CANDIDATE)

    digit_num = digit(N)
    path_env = env.get_current_env()
    for split in splits:
        split_imgs = []
        for asset_path, annotation_path in dr.item_paths(dataset_type=DatasetTypeDict[split]):
            idx += 1
            monitor.write_monitor_logger(percent=ymir_process_config['preprocess'] * idx / N)
            asset_path = osp.join(path_env.input.root_dir, path_env.input.assets_dir, asset_path)

            assert osp.exists(asset_path), f'cannot find {asset_path}'

            # valid data.yaml for training task
            # invalid data.yaml for infer and mining task
            if split in ['train', 'val']:
                annotation_path = osp.join(path_env.input.root_dir, path_env.input.annotations_dir, annotation_path)
                assert osp.exists(annotation_path), f'cannot find {annotation_path}'

                img_suffix = osp.splitext(asset_path)[1]
                img_path = osp.join(root_dir, 'images', str(idx).zfill(digit_num) + img_suffix)
                shutil.copy(asset_path, img_path)
                ann_path = osp.join(root_dir, 'labels', str(idx).zfill(digit_num) + '.txt')
                yolov5_ann_path = img2label_paths([img_path])[0]
                assert yolov5_ann_path == ann_path, f'bad yolov5_ann_path={yolov5_ann_path} and ann_path = {ann_path}'

                width, height = imagesize.get(img_path)
                with open(ann_path, 'w') as fw:
                    with open(annotation_path, 'r') as fr:
                        for line in fr.readlines():
                            class_id, xmin, ymin, xmax, ymax = [int(x) for x in line.strip().split(',')]

                            # class x_center y_center width height
                            # normalized xywh
                            # class_id 0-indexed
                            xc = (xmin + xmax) / 2 / width
                            yc = (ymin + ymax) / 2 / height
                            w = (xmax - xmin) / width
                            h = (ymax - ymin) / height
                            fw.write(f'{class_id} {xc} {yc} {w} {h}\n')

            split_imgs.append(img_path)
        with open(osp.join(root_dir, f'{split}.txt'), 'w') as fw:
            fw.write('\n'.join(split_imgs))

    # generate yaml
    config = env.get_executor_config()
    data = dict(path=root_dir,
                train="train.txt",
                val="val.txt",
                test='test.txt',
                nc=len(config['class_names']),
                names=config['class_names'])

    with open('data.yaml', 'w') as fw:
        fw.write(yaml.safe_dump(data))


def write_ymir_training_result(results, maps, rewrite=False):
    """
    results: (mp, mr, map50, map, loss)
    """
    if not rewrite:
        training_result_file = env.get_current_env().output.training_result_file
        if osp.exists(training_result_file):
            return 0

    model = env.get_universal_config()['model']
    class_names = env.get_universal_config()['class_names']
    map50 = maps

    # use `rw.write_training_result` to save training result
    rw.write_training_result(model_names=[f'{model}.yaml', 'best.pt', 'last.pt', 'best.onnx'],
                             mAP=float(np.mean(map50)),
                             classAPs={class_name: v
                                       for class_name, v in zip(class_names, map50.tolist())})
    return 0


if __name__ == '__main__':
    convert_ymir_to_yolov5('/out/yolov5_dataset')
