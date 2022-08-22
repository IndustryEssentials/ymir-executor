"""run.py:
img --(model)--> pred --(augmentation)--> (aug1_pred, aug2_pred, ..., augN_pred)
img --(augmentation)--> aug1_img --(model)--> pred1
img --(augmentation)--> aug2_img --(model)--> pred2
...
img --(augmentation)--> augN_img --(model)--> predN

dataload(img) --(model)--> pred
dataload(img, pred) --(augmentation1)--> (aug1_img, aug1_pred) --(model)--> pred1

1. split dataset with DDP sampler
2. use DDP model to infer sampled dataloader
3. gather infer result

"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as td
from functools import partial
from typing import List, Any
import cv2
from utils.augmentations import letterbox
import numpy as np
from ymir_exc.util import get_merged_config
from utils.ymir_yolov5 import YmirYolov5

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def load_image_file(img_file: str, img_size, stride):
    img = cv2.imread(img_file)
    img1 = letterbox(img, img_size, stride=stride, auto=True)[0]

    # preprocess: convert data format
    img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img1 = np.ascontiguousarray(img1)
    # img1 = torch.from_numpy(img1).to(self.device)

    img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
    # img1.unsqueeze_(dim=0)  # expand for batch dim
    return img1


class YmirDataset(td.Dataset):

    def __init__(self, images: List[str], annotations: List[Any] = None, augmentations=None, load_fn=None):
        super().__init__()
        self.annotations = annotations
        self.images = images
        self.augmentations = augmentations
        self.load_fn = load_fn

    def __getitem__(self, index):

        return self.load_fn(self.images[index])

    def __len__(self):
        return len(self.images)


def run(rank, size):
    """ Distributed function to be implemented later. """
    cfg = get_merged_config()
    model = YmirYolov5(cfg)

    load_fn = partial(load_image_file, img_size=model.img_size, stride=model.stride)

    with open(cfg.ymir.input.candidate_index_file, 'r') as f:
        images = [line.strip() for line in f.readlines()]

    # origin dataset
    origin_dataset = YmirDataset(images, load_fn=load_fn)

    sampler = None if rank == -1 else td.distributed.DistributedSampler(origin_dataset)
    origin_dataset_loader = td.Dataloader(origin_dataset,
                                          batch_size=4,
                                          shuffle=False,
                                          sampler=sampler,
                                          num_workers=0,
                                          pip_memory=True,
                                          drop_last=False)


    for batch in origin_dataset_loader:



def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
