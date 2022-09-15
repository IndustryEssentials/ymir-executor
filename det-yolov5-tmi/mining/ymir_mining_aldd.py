"""use fake DDP to infer
1. split data with `images_rank = images[RANK::WORLD_SIZE]`
2. infer on the origin dataset
3. infer on the augmentation dataset
4. save splited mining result with `torch.save(results, f'/out/mining_results_{RANK}.pt')`
5. merge mining result
"""
import os
import sys
import warnings
from functools import partial
from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as td
from easydict import EasyDict as edict
from mining.util import YmirDataset, load_image_file
from tqdm import tqdm
from utils.ymir_yolov5 import YmirYolov5
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class ALDD(object):
    def __init__(self, ymir_cfg: edict):
        avg_pool_kernel = 9
        max_pool_kernel = 30
        pad = (avg_pool_kernel - 1) // 2

        self.avg_pooling_layer = torch.nn.AvgPool2d(kernel_size=(avg_pool_kernel, avg_pool_kernel),
                                                    stride=(1, 1),
                                                    count_include_pad=False,
                                                    padding=(pad, pad))
        self.max_pooling_layer = torch.nn.MaxPool2d(kernel_size=(max_pool_kernel, max_pool_kernel),
                                                    stride=(30, 30),
                                                    padding=(2, 2))

        self.num_classes = len(ymir_cfg.param.class_names)
        if ymir_cfg.param.get('class_distribution_scores', ''):
            scores = [float(x.strip()) for x in ymir_cfg.param.class_distribution_scores.split(',')]
            if len(scores) < self.num_classes:
                warnings.warn('extend 1.0 to class_distribution_scores')
                scores.extend([1.0] * (self.num_classes - len(scores)))
            self.class_distribution_scores = np.array(scores[0:self.num_classes], dtype=np.float32)
        else:
            self.class_distribution_scores = np.array([1.0] * self.num_classes, dtype=np.float32)

    def calc_unc_val(self, heatmap: torch.Tensor) -> torch.Tensor:
        # mean of entropy
        prob_pixel = heatmap
        prob_pixel_m1 = 1 - heatmap
        ent = -(prob_pixel * torch.log(prob_pixel + 1e-12) + prob_pixel_m1 * torch.log(prob_pixel_m1 + 1e-12)
                )  # N, C, H, W
        ent = torch.sum(ent, dim=1, keepdim=True)  # N, 1, H, W
        mean_of_entropy = self.avg_pooling_layer(ent)  # N, 1, H, W

        # entropy of mean
        prob_local = self.avg_pooling_layer(heatmap)  # N, C, H, W
        prob_local_m1 = 1 - prob_local
        entropy_of_mean = -(
            prob_local * torch.log(prob_local + 1e-12) + prob_local_m1 * torch.log(prob_local_m1 + 1e-12))  # N, C, H, W
        entropy_of_mean = torch.sum(entropy_of_mean, dim=1, keepdim=True)  # N, 1, H, W

        uncertainty = entropy_of_mean - mean_of_entropy
        unc = self.max_pooling_layer(uncertainty)

        # aggregating
        scores = torch.mean(unc, dim=(1, 2, 3))
        return scores

    def compute_aldd_score(self, net_output: List[torch.Tensor], net_input_shape: Any):
        """
        args:
            imgs: list[np.array(H, W, C)]
        returns:
            scores: list of float
        """
        if not isinstance(net_input_shape, (list, tuple)):
            net_input_shape = (net_input_shape, net_input_shape)

        # CLASS_DISTRIBUTION_SCORE = np.array([1.0] * num_of_class)
        scores_list = []

        for feature_map in net_output:
            feature_map.sigmoid_()

        for each_class_index in range(self.num_classes):
            feature_map_list: List[torch.Tensor] = []

            # each_output_feature_map: [bs, 3, h, w, 5 + num_classes]
            for each_output_feature_map in net_output:
                net_output_conf = each_output_feature_map[:, :, :, :, 4]
                net_output_cls_mult_conf = net_output_conf * each_output_feature_map[:, :, :, :, 5 + each_class_index]
                # feature_map_reshape: [bs, 3, h, w]
                feature_map_reshape = torch.nn.functional.interpolate(net_output_cls_mult_conf,
                                                                      net_input_shape,
                                                                      mode='bilinear',
                                                                      align_corners=False)
                feature_map_list.append(feature_map_reshape)

            # len(net_output) = 3
            # feature_map_concate: [bs, 9, h, w]
            feature_map_concate = torch.cat(feature_map_list, 1)
            # scores: [bs, 1]
            scores = self.calc_unc_val(feature_map_concate)
            scores = scores.cpu().detach().numpy()
            scores_list.append(scores)

        # total_scores: [bs, num_classes]
        total_scores = np.array(scores_list)
        total_scores = total_scores * self.class_distribution_scores
        total_scores = np.sum(total_scores, axis=1)

        return total_scores


def run(ymir_cfg: edict, ymir_yolov5: YmirYolov5):
    # eg: gpu_id = 1,3,5,7  for LOCAL_RANK = 2, will use gpu 5.
    # gpu = int(ymir_yolov5.gpu_id.split(',')[LOCAL_RANK])
    gpu = LOCAL_RANK if LOCAL_RANK >= 0 else 0
    device = torch.device('cuda', gpu)
    ymir_yolov5.to(device)

    load_fn = partial(load_image_file, img_size=ymir_yolov5.img_size, stride=ymir_yolov5.stride)
    batch_size_per_gpu: int = ymir_yolov5.batch_size_per_gpu
    gpu_count: int = ymir_yolov5.gpu_count
    cpu_count: int = os.cpu_count() or 1
    num_workers_per_gpu = min([
        cpu_count // max(gpu_count, 1), batch_size_per_gpu if batch_size_per_gpu > 1 else 0,
        ymir_yolov5.num_workers_per_gpu
    ])

    with open(ymir_cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    # origin dataset
    if RANK != -1:
        images_rank = images[RANK::WORLD_SIZE]
    else:
        images_rank = images
    origin_dataset = YmirDataset(images_rank, load_fn=load_fn)
    origin_dataset_loader = td.DataLoader(origin_dataset,
                                          batch_size=batch_size_per_gpu,
                                          shuffle=False,
                                          sampler=None,
                                          num_workers=num_workers_per_gpu,
                                          pin_memory=ymir_yolov5.pin_memory,
                                          drop_last=False)

    mining_results = dict()
    dataset_size = len(images_rank)
    pbar = tqdm(origin_dataset_loader) if RANK == 0 else origin_dataset_loader
    miner = ALDD(ymir_cfg)
    for idx, batch in enumerate(pbar):
        with torch.no_grad():
            featuremap_output = ymir_yolov5.model.model(batch['image'].float().to(device))[1]
            unc_scores = miner.compute_aldd_score(featuremap_output, ymir_yolov5.img_size)

        for each_imgname, each_score in zip(batch["image_file"], unc_scores):
            mining_results[each_imgname] = each_score

        if RANK in [-1, 0]:
            ymir_yolov5.write_monitor_logger(stage=YmirStage.TASK, p=idx * batch_size_per_gpu / dataset_size)

    torch.save(mining_results, f'/out/mining_results_{RANK}.pt')


def main() -> int:
    ymir_cfg = get_merged_config()
    # note select_device(gpu_id) will set os.environ['CUDA_VISIBLE_DEVICES'] to gpu_id
    ymir_yolov5 = YmirYolov5(ymir_cfg, task='mining')

    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        # gpu = int(ymir_yolov5.gpu_id.split(',')[LOCAL_RANK])
        gpu = LOCAL_RANK if LOCAL_RANK >= 0 else 0
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    run(ymir_cfg, ymir_yolov5)

    # wait all process to save the mining result
    if WORLD_SIZE > 1:
        dist.barrier()

    if RANK in [0, -1]:
        results = []
        for rank in range(WORLD_SIZE):
            results.append(torch.load(f'/out/mining_results_{rank}.pt'))

        ymir_mining_result = []
        for result in results:
            for img_file, score in result.items():
                ymir_mining_result.append((img_file, score))
        rw.write_mining_result(mining_result=ymir_mining_result)

    print(f'rank: {RANK}, start destroy process group')
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
