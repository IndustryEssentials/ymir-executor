from typing import DefaultDict
from active_learning.model_inference import CenterNet
from active_learning.dataset import DataReader
import numpy as np
import cv2
import os


# generate result
# def mosaic(img_pool, img_paths):
#     image_shape = [img.shape[:2] for img in img_pool]
#     max_h = max(image_shape[0][0] + image_shape[2][0], image_shape[1][0] + image_shape[3][0])
#     max_w = max(image_shape[0][1] + image_shape[1][1], image_shape[2][1] + image_shape[3][1])
#     image = np.zeros((int(max_h), int(max_w), 3), dtype=np.uint8)
#     image[:image_shape[0][0], :image_shape[0][1], :] = img_pool[0]
#     image[:image_shape[1][0], image_shape[0][1]: image_shape[0][1] + image_shape[1][1], :] = img_pool[1]
#     image[max_h - image_shape[2][0]:, :image_shape[2][1], :] = img_pool[2]
#     image[max_h - image_shape[3][0]:, max_w - image_shape[3][1]:, :] = img_pool[3]
#     return image


# weight_file = './mobilenet_sc_cpu_combined_V1.4-0130.params'
# classes_file = './combined_class.txt'
# gpu_id = '5'
# confidence_thresh = 0.1
# batch_size = 1
# nms_thresh = 0.45
# input_dim = 512
# output_dim = 128
# net = CenterNet(
#     weight_file,
#     classes_file,
#     gpu_id,
#     confidence_thresh,
#     batch_size,
#     nms_thresh,
#     input_dim,
#     output_dim,
#     mode = 'combined'
# )


# with open("/data1/huangdewei/zengzhuoxi/modified_combined_wo_coco_TrainSet_V5.2.txt", 'r') as f:
#     lines = f.readlines()

# filtered_lines = list(filter(lambda x: x.find("imagenet") > 0, lines))
# # print(filtered_lines[0], filtered_lines[-1])
# # print(len(filtered_lines))
# data_reader = DataReader(filtered_lines)
# data_reader.start()

# img_pool = []
# img_paths = []
# count = 0
# while True:
#     img, img_path, stop = data_reader.dequeue()
#     img_pool.append(img)
#     img_paths.append(img_path)
#     preds, _, num_bbox = net.detect([img])
#     img_name = os.path.basename(img_path)
#     with open("imagenet_result/txt_result/" + img_name + ".txt", "w") as f:
#         for pred in preds[0]:
#             c = pred[0]
#             s = pred[1]
#             b = pred[2:6]
#             f.write("{} {} {} {} {} {}\n".format(int(c), float(s), int(b[0]), int(b[1]), int(b[2]), int(b[3])))
#     if count < 100:
#         net.draw_bbox(preds, num_bbox, [img_path], "imagenet_result/")
#     if len(img_pool) == 4:
#         print(count)
#         image = mosaic(img_pool, img_paths)
#         preds, _, num_bbox = net.detect([image])
#         if count < 100:
#             cv2.imwrite("imagenet_result/mosaic_image/mosaic_{}.jpg".format(count), image)
#             net.draw_bbox(preds, num_bbox, ["imagenet_result/mosaic_image/mosaic_{}.jpg".format(count)], "imagenet_result/")
#         with open("imagenet_result/txt_result/" + "mosaic_{}".format(count) + ".txt", "w") as f:
#             for pred in preds[0]:
#                 c = pred[0]
#                 s = pred[1]
#                 b = pred[2:6]
#                 f.write("{} {} {} {} {} {}\n".format(int(c), float(s), int(b[0]), int(b[1]), int(b[2]), int(b[3])))
#         img_pool = []
#         img_paths = []
#         count += 1
#     if stop:
#         break

# analyze result
# origin count defaultdict(<type 'int'>, {1: 1, 3: 31892, 4: 14401, 5: 6106, 6: 1955, 7: 405, 8: 35})
# mosaic count defaultdict(<type 'int'>, {1: 30264, 2: 10682, 3: 5940, 4: 3412, 5: 1914, 6: 850, 7: 262, 8: 45})
from collections import defaultdict
root_dir = "imagenet_result/txt_result"
origin_count = defaultdict(int)
mosaic_count = defaultdict(int)
for name in os.listdir(root_dir):
    p = os.path.join(root_dir, name)
    with open(p, "r") as f:
        lines = f.readlines()
    for line in lines:
        score = int(float(line.split(' ')[1]) * 10)
        if name.find("mosaic") >= 0:
            mosaic_count[score] += 1
        else:
            origin_count[score] += 1
print("origin count {}".format(origin_count))
print("mosaic count {}".format(mosaic_count))