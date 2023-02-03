#!/usr/bin/env python3

import datetime
import os.path as osp
from typing import Dict, List

import imagesize
import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm

import pycococreatortools

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "ymir",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [{
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
}]

CATEGORIES = [
    {
        'id': 1,
        'name': 'square',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'circle',
        'supercategory': 'shape',
    },
    {
        'id': 3,
        'name': 'triangle',
        'supercategory': 'shape',
    },
]


def convert(ymir_cfg: edict, results: List[Dict], with_blank_area: bool):
    """
    convert ymir infer result to coco instance segmentation format
    the mask is encode in compressed rle
    the is_crowd is True
    """
    class_names = ymir_cfg.param.class_names

    categories = []
    # categories should start from 0
    for idx, name in enumerate(class_names):
        categories.append(dict(id=idx, name=name, supercategory='none'))

    coco_output = {"info": INFO, "licenses": LICENSES, "categories": categories, "images": [], "annotations": []}

    image_id = 1
    annotation_id = 1

    for idx, d in enumerate(tqdm(results, desc='convert result to coco')):
        image_f = d['image']
        result = d['result']

        width, height = imagesize.get(image_f)

        image_info = pycococreatortools.create_image_info(image_id=image_id,
                                                          file_name=osp.basename(image_f),
                                                          image_size=(width, height))

        coco_output["images"].append(image_info)  # type: ignore

        # category_id === class_id start from 0
        unique_ids = np.unique(result)
        for np_class_id in unique_ids:
            if with_blank_area:
                class_id = int(np_class_id) - 1
            else:
                class_id = int(np_class_id)

            # remove background class in infer-result
            if with_blank_area and class_id < 0:
                continue

            assert class_id < len(class_names), f'class_id {class_id} must < class_num {len(class_names)}'
            category_info = {'id': class_id, 'is_crowd': True}
            binary_mask = result == np_class_id
            annotation_info = pycococreatortools.create_annotation_info(annotation_id,
                                                                        image_id,
                                                                        category_info,
                                                                        binary_mask,
                                                                        tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)  # type: ignore
                annotation_id = annotation_id + 1

        image_id += 1

    return coco_output
