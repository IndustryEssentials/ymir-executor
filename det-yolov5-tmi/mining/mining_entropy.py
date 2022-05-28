import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import tqdm
import yaml
from mmdet.apis import inference_detector, init_detector


def get_img_path(img_file):
    with open(img_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    names = [line.strip() for line in lines]
    return names


def preprocess_frcnn(src):
    img = cv2.resize(src, (1000, 600))
    return img


def forward_model(net, tmp_batch):
    features = []

    def forward_hook(module, input, output):
        features.append([output, input])

    hook_hanlder = net._modules.get("roi_head").bbox_head.register_forward_hook(forward_hook)
    inference_detector(net, tmp_batch)
    net_roi_output = features[0][0][0]
    softmax_func = torch.nn.Softmax(dim=-1)
    net_roi_output_with_sm = softmax_func(net_roi_output)  # apply softmax function to class output head
    roi_feature_dim, num_class_with_bg = net_roi_output_with_sm.shape

    # reshape tensor from batchx1000, num_cls_with_bg to batch, 1000, num_cls_with_bg
    net_roi_output_with_sm = torch.reshape(net_roi_output_with_sm, (-1, 1000, num_class_with_bg))
    hook_hanlder.remove()

    return net_roi_output_with_sm


def init_net(device):
    _net = init_detector("/in/model/model.py", "/in/model/model.pth", device)
    _net.eval()
    return _net


def decode_result(prediction):
    prediction = prediction.cpu().numpy()
    prediction = prediction[:, :, 1:]  # filter out bg class
    entropy_val_batch = []

    for each_prediction in prediction:
        max_conf = np.max(each_prediction, axis=1)
        mask = np.where(max_conf > 0.1)
        each_prediction = each_prediction[mask]
        if each_prediction.shape[0] == 0:
            entropy_val_batch.append(-10)
            continue
        entropy_val = (-each_prediction * np.log2(each_prediction)).sum(axis=1)
        max_entropy_val = np.max(entropy_val)
        entropy_val_batch.append(max_entropy_val)

    return entropy_val_batch


def detect_img(net, load_images):
    tmp_batch = list(map(preprocess_frcnn, load_images))

    batch_predict_result = forward_model(net, tmp_batch)
    post_process_result = decode_result(batch_predict_result)

    return post_process_result


def compute_cald_thread(net, img_names_batch):
    if len(img_names_batch) == 0:
        return []

    load_images = []
    for each_img_name in img_names_batch:
        img = cv2.imread(each_img_name)
        assert img is not None
        load_images.append(img)

    _score_list = detect_img(net, load_images)

    return _score_list


if __name__ == "__main__":

    config_file = "/in/config.yaml"
    with open(config_file, 'r', encoding="utf8") as f:
        config = yaml.safe_load(f)
    task_id = config["task_id"]
    gpu_id = config["gpu_id"]
    gpu_id = gpu_id.split(",")
    batch_size_per_gpu = config["batch_size_per_gpu"]
    batch_size = batch_size_per_gpu * len(gpu_id)

    imlist = get_img_path("/in/candidate/index.tsv")
    net_gpus = []
    devices = []
    for each_gpu_id in gpu_id:
        print(each_gpu_id)
        each_device = torch.device("cuda:{}".format(each_gpu_id))
        devices.append(each_device)
    for each_device in devices:
        each_net = init_net(each_device)
        net_gpus.append(each_net)

    OUTPUT_FILE_NAME = "/out/result.tsv"
    OUTPUT_FILE_NAME_TMP = "/out/tmp_result.txt"
    if os.path.isfile(OUTPUT_FILE_NAME_TMP):
        os.system("rm {}".format(OUTPUT_FILE_NAME_TMP))

    if not imlist:
        raise ValueError("imlist read fail")
    leftover = 0
    if len(imlist) % batch_size:
        leftover = 1
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))] for i in range(num_batches)]
    path2score = []
    print("********** start **********")
    for i, batch in tqdm.tqdm(enumerate(im_batches), total=len(im_batches)):
        log_handle = open("/out/monitor.txt", 'w')
        batches = [batch[k:k + batch_size_per_gpu] for k in range(0, len(batch), batch_size_per_gpu)]
        executor = ThreadPoolExecutor(max_workers=len(gpu_id))
        score_list = []

        # compute_cald_thread(net_gpus[0], batches[0])
        for result in executor.map(compute_cald_thread, net_gpus, batches):
            score_list += result
        with open(OUTPUT_FILE_NAME_TMP, 'a+') as opened_f:
            for each_name, each_score in zip(batch, score_list):
                output_txt = "{} {}\n".format(each_name, each_score)
                path2score.append([each_name, each_score])
                opened_f.write(output_txt)
        progress = float(i / num_batches)
        output_log_str = "{}\t{:.6f}\t{}\t{}\n".format(task_id, time.time(), progress, 2)
        log_handle.write(output_log_str)
    path2score.sort(key=lambda x: x[1], reverse=True)
    with open(OUTPUT_FILE_NAME_TMP, "w") as f:
        path2score = ["\t".join(list(map(str, x))) + "\n" for x in path2score]
        f.writelines(path2score)
    assert os.path.isfile(OUTPUT_FILE_NAME_TMP)
    os.system("mv {} {}".format(OUTPUT_FILE_NAME_TMP, OUTPUT_FILE_NAME))
