import os
import sys
sys.path.insert(0, "/data/xiongzihua/active_learning")
import cv2
import numpy as np
from active_learning.strategy import ALDD
from active_learning.model_inference import CenterNet
from active_learning.dataset import DataReader


def draw_heatmap(img, pred, img_name, heatmap_name, save_root, trans, norm_type="minmax"):
    h, w, c = img.shape
    save_name = img_name + "_" + heatmap_name + ".jpg"
    save_path = os.path.join(save_root, save_name)
    heatmap = None
    if norm_type == "minmax":
        heatmap = cv2.normalize(pred, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        heatmap = pred * norm_type
        heatmap = heatmap.astype(np.uint8)
    # print(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap = cv2.resize(heatmap, (w, h))
    heatmap = cv2.warpAffine(heatmap, trans, (w, h), flags = cv2.INTER_LINEAR)
    heatmap = cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)
    cv2.imwrite(save_path, heatmap)


img_list_path = "train_data_path/aldd_select_labeled_iter3_2000.txt"
with open(img_list_path, 'r') as f:
    img_list = f.readlines()
img_list = img_list[:30]
model_params_path = "../centernet-mx/deploy_model/mobilenet_sc_cpu_combined_aldd_select_iter2_2000-0130.params"
gpu_id = '6'
save_root = "./aldd_heatmap"
if not os.path.isdir(save_root):
    os.makedirs(save_root)
model = CenterNet(model_params_path, gpu_id=gpu_id)
aldd = ALDD(model=model)
datareader = DataReader(img_list, num_workers=8)
datareader.start()

while True:
    img, img_path, stop = datareader.dequeue()
    h, w, c = img.shape
    img_name = os.path.basename(img_path).split('.')[0]
    imgs = [img]
    heatmap, mean_of_entropy, entropy_of_mean, uncertainty, scores, c_batch, s_batch = aldd.compute_score(imgs, return_vis=True)
    trans = model.get_affine_transform(c_batch[0], s_batch[0], 0, [128, 128], inv = 1)
    uncertainty = uncertainty.asnumpy().reshape(128, 128)
    mean_of_entropy = mean_of_entropy.asnumpy().reshape(128, 128)
    entropy_of_mean = entropy_of_mean.asnumpy().reshape(128, 128)
    heatmap = heatmap.asnumpy().max(axis=1).reshape(128, 128)

    result, num_bbox = model.detect([img])
    img = model.draw_bbox(result, num_bbox, [img_path], "result/")
    draw_heatmap(img, heatmap, img_name, "heatmap", save_root, trans, 255)
    draw_heatmap(img, uncertainty, img_name, "uncertainty", save_root, trans, 255)
    draw_heatmap(img, mean_of_entropy, img_name, "e_mean_of_entropy", save_root, trans, 255)
    draw_heatmap(img, entropy_of_mean, img_name, "entropy_of_mean", save_root, trans, 255)
    # print(img.shape)
    # print(uncertainty.shape)
    # print(c_batch)
    if stop:
        break