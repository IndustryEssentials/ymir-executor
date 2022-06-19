# Copyright (c) OpenMMLab voc.py. All rights reserved.
# wangjiaxin 2022-04-25

from collections import OrderedDict
import os.path as osp

# from PIL import Image
import imagesize

import json
from .builder import DATASETS
from .api_wrappers import COCO
from .coco import CocoDataset

@DATASETS.register_module()
class YmirDataset(CocoDataset):
    """
    converted dataset by ymir system 1.0.0
    /in/assets: image files directory
    /in/annotations: annotation files directory
    /in/train-index.tsv: image_file \t annotation_file
    /in/val-index.tsv: image_file \t annotation_file
    """
    def __init__(self,
                 min_size=0,
                 ann_prefix='annotations',
                 **kwargs):
        self.min_size=min_size
        self.ann_prefix=ann_prefix
        super(YmirDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from TXT style ann_file.

        Args:
            ann_file (str): Path of TXT file.

        Returns:
            list[dict]: Annotation info from TXT file.
        """

        images = []
        categories = []
        # category_id is from 1 for coco, not 0
        for i, name in enumerate(self.CLASSES):
            categories.append({'supercategory':'none',
                              'id': i+1,
                              'name': name})

        annotations = []
        instance_counter = 1
        image_counter = 1

        with open(ann_file,'r') as fp:
            lines=fp.readlines()

        for line in lines:
            # split any white space
            img_path, ann_path = line.strip().split()
            img_path = osp.join(self.data_root, self.img_prefix, img_path)
            ann_path = osp.join(self.data_root, self.ann_prefix, ann_path)
            # img = Image.open(img_path)
            # width, height = img.size
            width, height = imagesize.get(img_path)
            images.append(
                dict(id=image_counter,
                     file_name=img_path,
                     ann_path=ann_path,
                     width=width,
                     height=height))

            try:
                anns = self.get_txt_ann_info(ann_path)
            except Exception as e:
                print(f'bad annotation for {ann_path} with {e}')
                anns = []

            for ann in anns:
                ann['image_id']=image_counter
                ann['id']=instance_counter
                annotations.append(ann)
                instance_counter+=1

            image_counter+=1

        ### pycocotool coco init
        self.coco = COCO()
        self.coco.dataset['type']='instances'
        self.coco.dataset['categories']=categories
        self.coco.dataset['images']=images
        self.coco.dataset['annotations']=annotations
        self.coco.createIndex()

        ### mmdetection coco init
        # avoid the filter problem in CocoDataset, view coco_api.py for detail
        self.coco.img_ann_map = self.coco.imgToAnns
        self.coco.cat_img_map = self.coco.catToImgs

        # get valid category_id (in annotation, start from 1, arbitary)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        # convert category_id to label(train_id, start from 0)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        # self.img_ids = list(self.coco.imgs.keys())
        assert len(self.img_ids) > 0, 'image number must > 0'
        N=len(self.img_ids)
        print(f'load {N} image from YMIR dataset')

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def dump(self, ann_file):
        with open(ann_file,'w') as fp:
            json.dump(self.coco.dataset, fp)

    def get_ann_path_from_img_path(self,img_path):
        img_id=osp.splitext(osp.basename(img_path))[0]
        return osp.join(self.data_root, self.ann_prefix, img_id+'.txt')

    def get_txt_ann_info(self, txt_path):
        """Get annotation from TXT file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        # img_id = self.data_infos[idx]['id']
        # txt_path = osp.splitext(img_path)[0]+'.txt'
        # txt_path = self.get_ann_path_from_img_path(img_path)
        anns = []
        if osp.exists(txt_path):
            with open(txt_path,'r') as fp:
                lines=fp.readlines()
        else:
            lines=[]
        for line in lines:
            obj=[int(x) for x in line.strip().split(',')[0:5]]
            # YMIR category id starts from 0, coco from 1
            category_id, xmin, ymin, xmax, ymax = obj
            bbox = [xmin, ymin, xmax, ymax]
            h,w=ymax-ymin,xmax-xmin
            ignore = 0
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = 1

            ann = dict(
                segmentation=[[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
                area=w*h,
                iscrowd=0,
                image_id=None,
                bbox=[xmin, ymin, w, h],
                category_id=category_id+1, # category id is from 1 for coco
                id=None,
                ignore=ignore
            )
            anns.append(ann)
        return anns

    def get_cat_ids(self, idx):
        """Get category ids in TXT file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        # img_path = self.data_infos[idx]['file_name']
        # txt_path = self.get_ann_path_from_img_path(img_path)
        txt_path = self.data_infos[idx]['ann_path']
        txt_path = osp.join(self.data_root, self.ann_prefix, txt_path)
        if osp.exists(txt_path):
            with open(txt_path,'r') as fp:
                lines = fp.readlines()
        else:
            lines = []

        for line in lines:
            obj = [int(x) for x in line.strip().split(',')]
            # label, xmin, ymin, xmax, ymax = obj
            cat_ids.append(obj[0])

        return cat_ids
