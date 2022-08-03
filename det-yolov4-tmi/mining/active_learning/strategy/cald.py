from mxnet import nd
import os
import scipy
import numpy as np

from .data_augment import horizontal_flip, cutout, rotate, intersect
from active_learning.utils import softmax


class CALD:
    """
    implement CALD, Consistency-basd Active Learning for Object Detection
    """
    def __init__(
        self, model=None, labeled_dataset=None
    ):
        self.model = model
        self.labeled_dataset = labeled_dataset
        self.beta = 1.3

    def _ious(self, boxes1, boxes2):
        """
        args:
            boxes1: np.array, (N, 4)
            boxes2: np.array, (M, 4)
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

    def compute_score(self, imgs, return_vis=False):
        """
        args:
            imgs: list[np.array(H, W, C)]
        returns:
            consistency, cls_js: float, float
            consistency(smaller indicates more informative)
            cls_js(larger means having different class distributions from labeled pool)
        """
        assert len(imgs) == 1, "CALD currently only support len(imgs) == 1"
        img = imgs[0]
        cls, max_scores, boxes, quality, cls_scores, cls_max_score, detect_flag = self.model.get_cald_input([img])
        origin_output = cls, max_scores, boxes, quality, cls_scores, cls_max_score

        if not detect_flag:
            return [[float("-inf"), float("-inf")], ]

        flip_img, flip_boxes = horizontal_flip(img, boxes)
        flip_output = self.model.get_cald_input([flip_img])
        detect_flag = detect_flag and flip_output[-1]

        rotate_img, rotate_boxes = rotate(img, boxes)
        rotate_output = self.model.get_cald_input([rotate_img])
        detect_flag = detect_flag and rotate_output[-1]

        cutout_img, cutout_boxes = cutout(img, boxes)
        cutout_output = self.model.get_cald_input([cutout_img])
        detect_flag = detect_flag and cutout_output[-1]
        if not detect_flag:
            return [[float("-inf"), float("-inf")], ]

        input_list = [flip_output, rotate_output, cutout_output]
        input_aug_boxes = [flip_boxes, rotate_boxes, cutout_boxes]
        consistency = 0
        for aug_output, origin_aug_boxes in zip(input_list, input_aug_boxes):
            consistency += self.get_consistency_per_aug(origin_output, aug_output, origin_aug_boxes)
        consistency /= float(len(input_list))

        pred_cls_distribution = self.get_pred_cls_distribution(cls_scores, input_list)
        if self.labeled_dataset is None:
            cls_js = 0.
        else:
            labeled_cls_distribution = self.labeled_dataset.get_cls_distribution()
            cls_js = self.get_clskl(pred_cls_distribution, labeled_cls_distribution)

        # score high means more informative
        score = -float(consistency)
        if return_vis:
            return [[score, cls_js], ], [flip_img, rotate_img, cutout_img], input_list, input_aug_boxes, cls, max_scores
        else:
            return [[score, cls_js], ]

    def get_pred_cls_distribution(self, cls_scores, input_list):
        aug_cls_scores = [output[4] for output in input_list]
        aug_cls_scores = [np.max(x, axis=0) for x in aug_cls_scores]
        cls_scores = np.max(cls_scores, axis=0)
        for score in aug_cls_scores:
            cls_scores += score
        cls_scores = softmax(cls_scores, axis=0)
        return cls_scores

    def get_clskl(self, pred_cls_distribution, labeled_cls_distribution):
        p = pred_cls_distribution
        q = labeled_cls_distribution
        m = (p + q) / 2.
        js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
        return js

    def get_consistency_per_aug(self, origin_output, aug_output, origin_aug_boxes):
        if len(origin_output[0]) == 0:
            return 2
        ious = self._ious(aug_output[2], origin_aug_boxes)  # N, M
        origin_idxs = np.argmax(ious, axis=1)
        consistency_per_aug = np.float("inf")
        for aug_idx, origin_idx in enumerate(origin_idxs):
            iou = ious[aug_idx, origin_idx]
            p = aug_output[4][aug_idx]
            q = origin_output[4][origin_idx]
            m = (p + q) / 2.
            js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
            if js < 0:
                js = 0
            consistency_box = iou
            consistency_cls = 0.5 * (origin_output[1][origin_idx] + aug_output[1][aug_idx]) * (1 - js)
            consistency_per_inst = abs(consistency_box + consistency_cls - self.beta)
            consistency_per_aug = min(consistency_per_aug, consistency_per_inst)

        return consistency_per_aug


if __name__ == '__main__':
    from active_learning.dataset import LabeledDataset
    from active_learning.model_inference import CenterNet
    import cv2

    weight_file = '../centernet-mx/deploy_model/mobilenet_sc_cpu_combined_aldd_select_iter2_2000-0130.params'
    classes_file = './combined_class.txt'
    gpu_id = '7'
    confidence_thresh = 0.1
    batch_size = 1
    nms_thresh = 0.45
    input_dim = 512
    output_dim = 128
    net = CenterNet(
        weight_file,
        classes_file,
        gpu_id,
        confidence_thresh,
        batch_size,
        nms_thresh,
        input_dim,
        output_dim,
        mode = 'combined'
    )

    labeled_dataset = LabeledDataset("train_data_path/combined_al_train_base.txt")
    img_path = "/data/zengzhuoxi/UbiquitousDetector/MultiDetector_38/own_test/own_TestSet/ordinary_hour_day_without_rain_zebra_crossing_20190415_162616_6784.jpg"
    img = cv2.imread(img_path)
    cald = CALD(net, labeled_dataset)
    boxes1 = np.array([[0,0,2,2], [2,0,4,2]])
    boxes2 = np.array([[1,1,3,3]])
    iou = cald._ious(boxes1, boxes2)
    print(iou)
    score = cald.compute_score([img])
    print(score)
    print(labeled_dataset.get_cls_distribution())
    print("done")
