import os
import sys
sys.path.insert(0, "/data/xiongzihua/active_learning")
import cv2
import numpy as np
from active_learning.strategy import CALD
from active_learning.model_inference import CenterNet
from active_learning.dataset import DataReader


img_list_path = "./temp/cald_5w_add_1w_score.txt"
with open(img_list_path, 'r') as f:
    img_list = f.readlines()
img_list = [x.split(' ')[0] for x in img_list]
img_list = img_list[:30]
model_params_path = "../centernet-mx/deploy_model/mobilenet_sc_cpu_random_5w-0130.params"
gpu_id = '6'
save_root = "./cald_result"
if not os.path.isdir(save_root):
    os.makedirs(save_root)
model = CenterNet(model_params_path, gpu_id=gpu_id)
cald = CALD(model=model)
datareader = DataReader(img_list, num_workers=8)
datareader.start()
classes_file = "./combined_class.txt"
classes = list(map(lambda x: x.strip(), open(classes_file, 'r').read().split('\n')))


def draw_box(img, boxes1, clses1, scores1, boxes2, clses2, scores2, img_name, aug):
    for box, cls, score in zip(boxes1, clses1, scores1):
        xmin = int(box[0]) if int(box[0]) >= 0 else 0
        ymin = int(box[1]) if int(box[1]) >= 0 else 0
        xmax = int(box[2])
        ymax = int(box[3])
        c1 = tuple([xmin, ymin])
        c2 = tuple([xmax, ymax])
        color = (0, 0, 255)
        label = "origin_{0} {1:.3f}".format(classes[int(cls)], float(score))
        img = cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
        img = cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 7), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    for box, cls, score in zip(boxes2, clses2, scores2):
        xmin = int(box[0]) if int(box[0]) >= 0 else 0
        ymin = int(box[1]) if int(box[1]) >= 0 else 0
        xmax = int(box[2])
        ymax = int(box[3])
        c1 = tuple([xmin, ymin])
        c2 = tuple([xmax, ymax])
        color = (0, 255, 0)
        label = "{0} {1:.3f}".format(classes[int(cls)], float(score))
        img = cv2.rectangle(img, c1, c2, color, 2)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 7), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    cv2.imwrite(os.path.join(save_root, img_name + "_{}".format(aug) + ".jpg"), img)


while True:
    img, img_path, stop = datareader.dequeue()
    h, w, c = img.shape
    img_name = os.path.basename(img_path).split('.')[0]
    imgs = [img]
    scores, imgs, input_list, input_aug_boxes, cls, scores = cald.compute_score(imgs, return_vis=True)
    flip_img, rotate_img, cutout_img = imgs
    flip_output, rotate_output, cutout_output = input_list
    pred_flip_boxes, pred_flip_cls, pred_flip_scores = flip_output[2], flip_output[0], flip_output[1]
    pred_rotate_boxes, pred_rotate_cls, pred_rotate_scores = rotate_output[2], rotate_output[0], rotate_output[1]
    pred_cutout_boxes, pred_cutout_cls, pred_cutout_scores = cutout_output[2], cutout_output[0], cutout_output[1]
    flip_boxes, rotate_boxes, cutout_boxes = input_aug_boxes
    
    draw_box(flip_img, flip_boxes, cls, scores, pred_flip_boxes, pred_flip_cls, pred_flip_scores, img_name, "flip")
    draw_box(rotate_img, rotate_boxes, cls, scores,  pred_rotate_boxes, pred_rotate_cls, pred_rotate_scores, img_name, "rotate")
    draw_box(cutout_img, cutout_boxes, cls, scores, pred_cutout_boxes, pred_cutout_cls, pred_cutout_scores, img_name, "cutout")
    # print(img.shape)
    # print(uncertainty.shape)
    # print(c_batch)
    if stop:
        break