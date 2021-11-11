import shutil
import os
from collections import defaultdict
import numpy as np


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


class LabeledDataset:
    def __init__(self, img_list_path="labeled_data.txt", dst_img_list_path=None):
        self.img_list_path = img_list_path
        self.dst_img_list_path = dst_img_list_path
        if not os.path.exists(self.img_list_path):
            with open(self.img_list_path, 'w') as f:
                pass
        with open(self.img_list_path, 'r') as f:
            self.img_list = f.readlines()
        self.num_samples = len(self.img_list)
        self.cls_distribution = None
        # # backup
        # prefix, suffix = self.img_list_path.split('.')
        # backup_path = prefix + "_al_backup." + suffix
        # shutil.copyfile(self.img_list_path, backup_path)

    def get_cls_distribution(self):
        """
        this is write for combined detection dataset. it means
        xxyyzz.txt in same folder as xxyyzz.jpg, and xxyyzz.txt has following type
        cls_idx x1 y1 x2 y2
        """
        if self.cls_distribution is not None:
            return self.cls_distribution
        # count cls
        print("generate cls distribution")
        cls_count = defaultdict(int)
        for img_path in self.img_list:
            img_path = img_path.strip()
            txt_path = img_path.split('.')[0] + ".txt"
            with open(txt_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                cls_idx = int(line.split(' ')[0])
                cls_count[cls_idx] += 1
        cls_count = [cls_count[k] for k in sorted(cls_count.keys())]
        cls_distribution = np.array(cls_count)
        cls_distribution = softmax(cls_distribution, axis=0)
        self.cls_distribution = cls_distribution
        return self.cls_distribution

    def merge(self, added_img_list):
        #  update(add) labeled dataset
        shutil.copyfile(self.img_list_path, self.dst_img_list_path)
        with open(self.dst_img_list_path, 'a') as f:
            f.writelines(added_img_list)