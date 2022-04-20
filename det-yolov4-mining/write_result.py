from collections import defaultdict
import json
import logging
import os
import sys
import time
from typing import List

import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import tqdm

import monitor_process
from active_learning.utils import log_collector, TaskState

# image_name = 0
# os.environ["VISIBLE_CUDA_DEVICES"] = "2"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"


infer_task_id = 'default-infer-task'


def _convert_square(bbox, exth_scale=0.0, extw_scale=0.0):
    """
    Paramters:
    bbox:numpy array of [x1,y1,x2,y2]
    exth_scale: equally height extension ratio for face padding
    extw_sacla: equally width extension ratio for face padding
    Returns:
    the squared and extened bounding box, it may contain coordinates which outside the image
    """
    if bbox is None:
        return None

    sbbox = bbox.copy()
    sbbox[:, 0] = bbox[:, 0]
    sbbox[:, 1] = bbox[:, 1]
    sbbox[:, 2] = bbox[:, 2]
    sbbox[:, 3] = bbox[:, 3]

    return sbbox


def try_gpu(num_list) -> list:
    '''if GPU is available, return mx.gpu(0);else return mx.cpu()'''
    ctx = []
    for num in num_list:
        try:
            tmp_ctx = mx.gpu(int(num))
            _ = nd.array([0], ctx=tmp_ctx)
            ctx.append(tmp_ctx)
        except Exception as e:
            logging.error("gpu {}:".format(num), e)
    if not ctx:
        ctx.append(mx.cpu())
    return ctx


def resize_image(im, w, h):
    # im [h, w, c]
    im_h, im_w, im_c = im.shape
    if im_w == w and im_h == h:
        return np.copy(im)

    resized = np.zeros([h, w, im_c])
    part = np.zeros([im_h, w, im_c])

    w_scale = (float)(im_w - 1) / (w - 1)
    h_scale = (float)(im_h - 1) / (h - 1)

    for k in range(im_c):
        for r in range(im_h):
            for c in range(w):
                if c == w - 1 or im_w == 1:
                    val = im[r, im_w - 1, k]
                else:
                    sx = c * w_scale
                    ix = int(sx)
                    dx = sx - ix
                    val = (1 - dx) * im[r, ix, k] + dx * im[r, ix + 1, k]
                part[r, c, k] = val

    for k in range(im_c):
        for r in range(h):
            sy = r * h_scale
            iy = int(sy)
            dy = sy - iy
            for c in range(w):
                val = (1 - dy) * part[iy, c, k]
                resized[r, c, k] = val
            if r == h - 1 or im_h == 1:
                continue
            for c in range(w):
                val = dy * part[iy + 1, c, k]
                resized[r, c, k] += val

    return resized


def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas


def _prep_image(img, inp_dim):
    img = cv2.resize(img, (inp_dim, inp_dim), cv2.INTER_NEAREST)
    img = np.transpose(img[:, :, ::-1], (2, 0, 1)).astype("float32")
    img /= 255.0
    return img


def _predict_transform(prediction, input_dim, anchors, ctx):
    if not isinstance(anchors, nd.NDArray):
        anchors = nd.array(anchors, ctx=ctx)
    batch_size = prediction.shape[0]
    strides = [input_dim // 32, input_dim // 32 * 2, input_dim // 32 * 2 * 2]

    if len(anchors) == 9:
        anchors_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        step = [[0, strides[0]**2 * 3], [strides[0]**2 * 3, strides[0]**2 * 3 + strides[1]**2 * 3],
                [strides[0]**2 * 3 + strides[1]**2 * 3, strides[0]**2 * 3 + strides[1]**2 * 3 + strides[2]**2 * 3]]
    else:
        raise RuntimeError(f"invalid anchors: {anchors}, should have length 9")

    boxes = len(anchors) // 3
    for i in range(3):
        stride = strides[i]
        grid = np.arange(stride)
        a, b = np.meshgrid(grid, grid)
        x_offset = nd.array(a.reshape((-1, 1)), ctx=ctx)
        y_offset = nd.array(b.reshape((-1, 1)), ctx=ctx)
        x_y_offset = nd.repeat(nd.expand_dims(
            nd.repeat(nd.concat(x_offset, y_offset, dim=1), repeats=boxes, axis=0).reshape((-1, 2)), 0),
                               repeats=batch_size,
                               axis=0)
        temp_anchors = nd.repeat(nd.expand_dims(
            nd.repeat(nd.expand_dims(anchors[anchors_masks[i]], 0), repeats=stride * stride, axis=0).reshape((-1, 2)),
            0),
                                 repeats=batch_size,
                                 axis=0)
        prediction[:, step[i][0]:step[i][1], :2] = (prediction[:, step[i][0]:step[i][1], :2] + x_y_offset)
        prediction[:, step[i][0]:step[i][1], :2] *= (float(input_dim) / stride)
        prediction[:, step[i][0]:step[i][1], 2:4] = nd.exp(prediction[:, step[i][0]:step[i][1], 2:4]) * temp_anchors
    return prediction


def bbox_iou(box1, box2, transform=True):
    if not isinstance(box1, nd.NDArray):
        box1 = nd.array(box1)
    if not isinstance(box2, nd.NDArray):
        box2 = nd.array(box2)
    box1 = nd.abs(box1)
    box2 = nd.abs(box2)
    if transform:
        tmp_box1 = box1.copy()
        tmp_box1[:, 0] = box1[:, 0] - box1[:, 2] / 2.0
        tmp_box1[:, 1] = box1[:, 1] - box1[:, 3] / 2.0
        tmp_box1[:, 2] = box1[:, 0] + box1[:, 2] / 2.0
        tmp_box1[:, 3] = box1[:, 1] + box1[:, 3] / 2.0

        tmp_box2 = box2.copy()
        tmp_box2[:, 0] = box2[:, 0] - box2[:, 2] / 2.0
        tmp_box2[:, 1] = box2[:, 1] - box2[:, 3] / 2.0
        tmp_box2[:, 2] = box2[:, 0] + box2[:, 2] / 2.0
        tmp_box2[:, 3] = box2[:, 1] + box2[:, 3] / 2.0

    # Get the coordinates of bounding box
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of intersection rectangle
    inter_rect_x1 = nd.where(b1_x1 > b2_x1, b1_x1, b2_x1)
    inter_rect_y1 = nd.where(b1_y1 > b2_y1, b1_y1, b2_y1)
    inter_rect_x2 = nd.where(b1_x2 < b2_x2, b1_x2, b2_x2)
    inter_rect_y2 = nd.where(b1_y2 < b2_y2, b1_y2, b2_y2)

    # Intersection Area
    inter_area = nd.clip(inter_rect_x2 - inter_rect_x1 + 1, a_min=0, a_max=1000) * nd.clip(
        inter_rect_y2 - inter_rect_y1 + 1, a_min=0, a_max=1000)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area)
    # iou[inter_area >= b1_area] = 0.8
    # iou[inter_area >= b2_area] = 0.8

    return nd.clip(iou, 1e-5, 1. - 1e-5)


def _write_results(prediction, num_classes, input_dim, confidence=0.5, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).expand_dims(2)
    prediction = prediction * conf_mask
    batch_size = prediction.shape[0]
    box_corner = nd.zeros(prediction.shape, dtype="float32")
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    # xyxy
    output = None
    prediction = prediction.asnumpy()

    for ind in range(batch_size):
        image_pred = prediction[ind]
        each_batch_bboxes_conf = image_pred[:, 4]
        for cls in range(num_classes):
            image_pred[:, 5 + cls] = image_pred[:, 5 + cls] * each_batch_bboxes_conf
            conf_mask = (image_pred[:, 5 + cls] > confidence)
            image_pred[:, 5 + cls] = image_pred[:, 5 + cls] * conf_mask
            no_zero_ind = np.nonzero(image_pred[:, 5 + cls])[0]
            num_samples = no_zero_ind.shape[0]
            if num_samples == 0:
                continue
            image_pred_ = image_pred[no_zero_ind, :]
            # image_pred_[:, [1, 3]] -= 70.28125
            # image_pred_[:, :4] /= 0.40625
            class_id = np.expand_dims(np.array([cls] * num_samples), 1)
            box_objness = np.expand_dims(image_pred_[:, 5 + cls], 1)
            image_pred_ = np.concatenate([class_id, box_objness, image_pred_[:, :4]], axis=1)
            image_pred_[:, 2:6] = np.clip(image_pred_[:, 2:6], 0, input_dim)
            image_pred_after_nms = nd.contrib.box_nms(nd.array(image_pred_), overlap_thresh=nms_conf)
            image_pred_after_nms = image_pred_after_nms.asnumpy()
            # find box that after nms
            conf_mask = (image_pred_after_nms[:, 0] != -1)
            pos_ind = np.nonzero(conf_mask)
            image_pred_after_nms = image_pred_after_nms[pos_ind]
            batch_ind = np.ones((image_pred_after_nms.shape[0], 1)) * ind
            seq = nd.concat(nd.array(batch_ind), nd.array(image_pred_after_nms), dim=1)
            if output is None:
                output = seq
            else:
                output = nd.concat(output, seq, dim=0)
    return output


# """
# yolov3 alexey
def _prep_results(load_images, img_batch, output, input_dim):
    im_dim_list = nd.array([(x.shape[1], x.shape[0]) for x in load_images])
    im_dim_list = nd.tile(im_dim_list, 2)
    im_dim_list = im_dim_list[output[:, 0], :]
    scaling_factor = input_dim / im_dim_list
    output[:, 3:7] /= scaling_factor
    for i in range(output.shape[0]):
        output[i, [3, 5]] = nd.clip(output[i, [3, 5]], a_min=0.0, a_max=im_dim_list[i][0].asscalar())
        output[i, [4, 6]] = nd.clip(output[i, [4, 6]], a_min=0.0, a_max=im_dim_list[i][1].asscalar())

    output = output.asnumpy()
    boxes = []
    for i in range(len(load_images)):
        bboxs = []
        for bbox in output:
            if i == int(bbox[0]):
                bboxs.append(bbox)
        for x in bboxs:  # image_pred: _, bbox, ness * max_prob, max_prob, max_cls (prob)
            cls = int(x[1])
            ness = float(x[2])
            boxes.append([img_batch[i], cls, ness, int(x[3]), int(x[4]), int(x[5] - x[3]), int(x[6] - x[4])])
    return boxes


class darknet_mxnet():
    def __init__(self,
                 weights_file,
                 class_names: List[str],
                 gpu_id: str = "0",
                 confidence_thresh: float = 0.2,
                 batch_size: int = 16,
                 nms_thresh: float = 0.45,
                 input_dim: int = 608,
                 anchors: str = None,
                 tiny_flag: bool = False):
        self.classes = class_names
        self.gpu = [int(x) for x in gpu_id.split(",")] if gpu_id else None
        self.ctx = try_gpu(self.gpu)[0] if self.gpu else mx.cpu()
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        model_file = weights_file.replace("0000", "symbol").replace(".params", ".json")
        self.net = gluon.SymbolBlock.imports(model_file, ['data'], weights_file, ctx=self.ctx)
        assert anchors is not None
        self.anchors = anchors
        self.tiny_flag = tiny_flag
        self.net.hybridize()

    def draw_bbox(self, boxes, dst_dir):
        img_dir = None
        img = None
        for i in range(len(boxes)):
            x = boxes[i]
            if x[0] != img_dir:
                if img_dir is not None:
                    cv2.imwrite(dst_dir + os.path.basename(img_dir), img)
                img_dir = x[0]
                img = cv2.imread(img_dir)
            c1 = tuple([int(x[3]), int(x[4])])
            c2 = tuple([int(x[5]) + int(x[3]), int(x[6]) + int(x[4])])
            cls = int(x[1])
            ness = float(x[2])
            color = [0, 0, 255]
            label = "{:} {:.5f}".format(self.classes[cls], ness)
            cv2.rectangle(img, c1, c2, color, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
            cv2.putText(img, label, (c1[0], c1[1] - t_size[1] + 7), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 2)
        if img_dir is not None:
            cv2.imwrite(dst_dir + os.path.basename(img_dir), img)

    def _save_result(self, boxes):
        save_dict = defaultdict(dict)
        for i in range(len(boxes)):
            x = boxes[i]
            class_index = int(x[1])
            ness = float(x[2])
            x_min = int(x[3])
            y_min = int(x[4])
            width = int(x[5])
            height = int(x[6])
            img_name = os.path.basename(x[0])

            save_content = {
                "box": {
                    'x': x_min,
                    'y': y_min,
                    'w': width,
                    'h': height
                },
                "class_name": self.classes[class_index],
                "score": ness
            }
            if img_name not in save_dict:
                save_dict[img_name]['annotations'] = []

            save_dict[img_name]['annotations'].append(save_content)

        return save_dict

    def detect(self, imlist: List[str]) -> dict:
        if not imlist:
            raise ValueError('no image to infer')

        logging.info(f"context: {self.ctx}, batch size: {self.batch_size}, image count: {len(imlist)}")

        leftover = 0
        if len(imlist) % self.batch_size:
            leftover = 1
        num_batches = len(imlist) // self.batch_size + leftover
        im_batches = [
            imlist[i * self.batch_size:min((i + 1) * self.batch_size, len(imlist))] for i in range(num_batches)
        ]
        boxes = None
        sc = 0
        for i, batch in tqdm.tqdm(enumerate(im_batches), total=len(im_batches)):
            batch_start = time.time()

            load_images = [cv2.imread(img) for img in batch]
            tmp_batch = list(map(_prep_image, load_images, [self.input_dim for x in range(len(batch))]))
            tmp_batch = nd.array(tmp_batch, ctx=self.ctx)

            if self.tiny_flag:
                raise RuntimeError("yolo tiny not supported")
            prediction = _predict_transform(self.net(tmp_batch), self.input_dim, self.anchors, self.ctx)
            prediction = _write_results(prediction,
                                        self.num_classes,
                                        confidence=self.confidence_thresh,
                                        nms_conf=self.nms_thresh,
                                        input_dim=self.input_dim)

            if prediction is not None:
                result = _prep_results(load_images, batch, prediction, input_dim=self.input_dim)
                if boxes is None:
                    boxes = result
                else:
                    boxes = np.concatenate([boxes, result], axis=0)
            else:
                logging.debug('No detection were made!')

            batch_end = time.time()
            cost = (batch_end - batch_start) / len(batch)
            sc += cost

            # write monitor process
            if i > 0 and i % 4 == 0:
                infer_percent = i / len(im_batches)
                monitor_process.infer_percent = infer_percent
                log_collector.monitor_collect(percent=monitor_process.get_total_percent(),
                                              status=TaskState.RUNNING)

        logging.debug(f"\naverage infer seconds per batch: {sc / len(im_batches)}")

        if boxes is not None:
            new_boxes = []
            for tmp in boxes:
                name, cls, score, *bbox_xy = tmp
                new_tmp = [int(float(ii)) for ii in bbox_xy]
                new_tmp[2] = new_tmp[2] + new_tmp[0]
                new_tmp[3] = new_tmp[3] + new_tmp[1]
                new_tmp = _convert_square(np.array([new_tmp]))
                new_boxes.append([
                    name, cls, score,
                    str(new_tmp[0][0]),
                    str(new_tmp[0][1]),
                    str(new_tmp[0][2] - new_tmp[0][0]),
                    str(new_tmp[0][3] - new_tmp[0][1])
                ])
            save_dict = self._save_result(new_boxes)
        else:
            save_dict = {}
        save_dict = {"detection": save_dict}
        return save_dict


def run(candidate_path: str, result_path: str, gpu_id: str, confidence_thresh: float, nms_thresh: float,
        image_width: int, image_height: int, model_params_path: str, anchors: str, class_names: List[str],
        batch_size: int) -> None:
    logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.INFO)

    start = time.time()

    assert image_height == image_width

    anchors = [int(each) for each in anchors.split(",")]
    anchors = np.array(anchors).reshape([-1, 2])

    with open(candidate_path, 'r') as f:
        imlist = f.readlines()
    imlist = [each.strip() for each in imlist]
    imlist = [os.path.join("/in/candidate", each) if not os.path.isabs(each) else each for each in imlist]

    net = darknet_mxnet(weights_file=model_params_path,
                        class_names=class_names,
                        gpu_id=gpu_id,
                        confidence_thresh=confidence_thresh,
                        batch_size=batch_size,
                        input_dim=image_height,
                        nms_thresh=nms_thresh,
                        anchors=anchors,
                        tiny_flag=False)
    save_dict = net.detect(imlist)

    with open(result_path, 'w') as f:
        json.dump(save_dict, f, indent=2)

    logging.debug(f"container time: {time.time() - start}")
