"""
Consistency-based Active Learning for Object Detection CVPR 2022 workshop
official code: https://github.com/we1pingyu/CALD/blob/master/cald_train.py
"""
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from easydict import EasyDict as edict
from nptyping import NDArray
from scipy.stats import entropy
from tqdm import tqdm
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, get_ymir_process

from mining.data_augment import cutout, horizontal_flip, intersect, resize, rotate
from utils.ymir_yolov5 import BBOX, CV_IMAGE, YmirYolov5


def split_result(result: NDArray) -> Tuple[BBOX, NDArray, NDArray]:
    if len(result) > 0:
        bboxes = result[:, :4].astype(np.int32)
        conf = result[:, 4]
        class_id = result[:, 5]
    else:
        bboxes = np.zeros(shape=(0, 4), dtype=np.int32)
        conf = np.zeros(shape=(0, 1), dtype=np.float32)
        class_id = np.zeros(shape=(0, 1), dtype=np.int32)

    return bboxes, conf, class_id


class MiningCald(YmirYolov5):

    def __init__(self, cfg: edict):
        super().__init__(cfg)

        if cfg.ymir.run_mining and cfg.ymir.run_infer:
            # multiple task, run mining first, infer later
            mining_task_idx = 0
            task_num = 2
        else:
            mining_task_idx = 0
            task_num = 1

        self.task_idx = mining_task_idx
        self.task_num = task_num

    def mining(self) -> List:
        N = dr.items_count(env.DatasetType.CANDIDATE)
        monitor_gap = max(1, N // 1000)
        idx = -1
        beta = 1.3
        mining_result = []
        for asset_path, _ in tqdm(dr.item_paths(dataset_type=env.DatasetType.CANDIDATE)):
            img = cv2.imread(asset_path)
            # xyxy,conf,cls
            result = self.predict(img)
            bboxes, conf, _ = split_result(result)
            if len(result) == 0:
                # no result for the image without augmentation
                mining_result.append((asset_path, -beta))
                continue

            consistency = 0.0
            aug_bboxes_dict, aug_results_dict = self.aug_predict(img, bboxes)
            for key in aug_results_dict:
                # no result for the image with augmentation f'{key}'
                if len(aug_results_dict[key]) == 0:
                    consistency += beta
                    continue

                bboxes_key, conf_key, _ = split_result(aug_results_dict[key])
                cls_scores_aug = 1 - conf_key
                cls_scores = 1 - conf

                consistency_per_aug = 2.0
                ious = get_ious(bboxes_key, aug_bboxes_dict[key])
                aug_idxs = np.argmax(ious, axis=0)
                for origin_idx, aug_idx in enumerate(aug_idxs):
                    max_iou = ious[aug_idx, origin_idx]
                    if max_iou == 0:
                        consistency_per_aug = min(consistency_per_aug, beta)
                    p = cls_scores_aug[aug_idx]
                    q = cls_scores[origin_idx]
                    m = (p + q) / 2.
                    js = 0.5 * entropy([p, 1 - p], [m, 1 - m]) + 0.5 * entropy([q, 1 - q], [m, 1 - m])
                    if js < 0:
                        js = 0
                    consistency_box = max_iou
                    consistency_cls = 0.5 * (conf[origin_idx] + conf_key[aug_idx]) * (1 - js)
                    consistency_per_inst = abs(consistency_box + consistency_cls - beta)
                    consistency_per_aug = min(consistency_per_aug, consistency_per_inst.item())

                    consistency += consistency_per_aug

            consistency /= len(aug_results_dict)

            mining_result.append((asset_path, consistency))
            idx += 1

            if idx % monitor_gap == 0:
                percent = get_ymir_process(stage=YmirStage.TASK,
                                           p=idx / N,
                                           task_idx=self.task_idx,
                                           task_num=self.task_num)
                monitor.write_monitor_logger(percent=percent)

        return mining_result

    def aug_predict(self, image: CV_IMAGE, bboxes: BBOX) -> Tuple[Dict[str, BBOX], Dict[str, NDArray]]:
        """
        for different augmentation methods: flip, cutout, rotate and resize
            augment the image and bbox and use model to predict them.

        return the predict result and augment bbox.
        """
        aug_dict = dict(flip=horizontal_flip, cutout=cutout, rotate=rotate, resize=resize)

        aug_bboxes = dict()
        aug_results = dict()
        for key in aug_dict:
            aug_img, aug_bbox = aug_dict[key](image, bboxes)

            aug_result = self.predict(aug_img)
            aug_bboxes[key] = aug_bbox
            aug_results[key] = aug_result

        return aug_bboxes, aug_results


def get_ious(boxes1: BBOX, boxes2: BBOX) -> NDArray:
    """
    args:
        boxes1: np.array, (N, 4), xyxy
        boxes2: np.array, (M, 4), xyxy
    return:
        iou: np.array, (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    iner_area = intersect(boxes1, boxes2)
    area1 = area1.reshape(-1, 1).repeat(area2.shape[0], axis=1)
    area2 = area2.reshape(1, -1).repeat(area1.shape[0], axis=0)
    iou = iner_area / (area1 + area2 - iner_area + 1e-14)
    return iou


def main():
    cfg = get_merged_config()
    miner = MiningCald(cfg)
    mining_result = miner.mining()
    rw.write_mining_result(mining_result=mining_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
