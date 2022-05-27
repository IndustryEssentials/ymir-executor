import os.path as osp
import numpy as np
from tqdm import tqdm
from mining.data_augment import horizontal_flip, cutout, rotate, intersect, resize
# from mmdet.apis import inference_detector, init_detector
from scipy.stats import entropy
import cv2
from executor import dataset_reader as dr, env, monitor, result_writer as rw

from loguru import logger
from utils.ymir_yolov5 import Ymir_Yolov5
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device

def init_detector(device):
    executor_config=env.get_executor_config()

    weights = None
    model_params_path = executor_config['model_params_path']
    if 'best.pt' in model_params_path:
        weights = '/in/models/best.pt'
    else:
        for f in model_params_path:
            if f.endswith('.pt'):
                weights=f'/in/models/{f}'
                break 
    
    if weights is None:
        weights = 'yolov5s.pt'
        logger.info(f'cannot find pytorch weight in {model_params_path}, use {weights} instead')

    model = DetectMultiBackend(weights=weights,
        device=device,
        dnn=False, # not use opencv dnn for onnx inference
        data='data.yaml') # dataset.yaml path

    return model

class MiningCald(Ymir_Yolov5):
    def mining(self):
        def split_result(result):
            if len(result)>0:
                bboxes=result[:,:4]
                conf=result[:,4]
                class_id=result[:,5]
            else:
                bboxes=[]
                conf=[]
                class_id=[]
            
            return bboxes, conf, class_id

        path_env = env.get_current_env()
        N=dr.items_count(env.DatasetType.CANDIDATE)
        idx=0
        beta=1.3
        mining_result=[]
        for asset_path, _ in tqdm(dr.item_paths(dataset_type=env.DatasetType.CANDIDATE)):
            img_path=osp.join(path_env.input.root_dir, path_env.input.assets_dir, asset_path)
            img = cv2.imread(img_path)
            # xyxy,conf,cls
            result = self.predict(img)
            if len(result)>0:
                bboxes = result[:,:4]
            else:
                bboxes = []
                mining_result.append((asset_path,-beta))
                continue 
            
            consistency=0
            aug_bboxes_dict, aug_results_dict = self.aug_predict(img, bboxes)
            bboxes, conf, class_id = split_result(result)
            for key in aug_results_dict:
                if len(aug_results_dict[key])==0:
                    consistency += beta
                    continue

                bboxes_key, conf_key, class_id_key = split_result(aug_results_dict[key])
                cls_scores_aug = 1 - conf_key 
                cls_scores = 1 - conf 

                consistency_per_aug=2
                ious = _ious(bboxes_key,aug_bboxes_dict[key])
                aug_idxs = np.argmax(ious, axis=0)
                for origin_idx, aug_idx in enumerate(aug_idxs):
                    iou = ious[aug_idx, origin_idx]
                    if iou == 0:
                        consistency_per_aug = min(consistency_per_aug, beta)
                    p = cls_scores_aug[aug_idx]
                    q = cls_scores[origin_idx]
                    m = (p + q) / 2.
                    js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
                    if js < 0:
                        js = 0
                    consistency_box = iou
                    consistency_cls = 0.5 * (conf[origin_idx] + conf_key[aug_idx]) * (1 - js)
                    consistency_per_inst = abs(consistency_box + consistency_cls - beta)
                    consistency_per_aug = min(consistency_per_aug, consistency_per_inst.item())

                    consistency += consistency_per_aug

            consistency /= len(aug_results_dict)
            
            mining_result.append((asset_path, consistency))
            idx+=1
            monitor.write_monitor_logger(percent=0.1 + 0.8*idx/N)

        return mining_result

    def aug_predict(self, image, bboxes):
        aug_dict=dict(flip=horizontal_flip,
            cutout=cutout,
            rotate=rotate,
            resize=resize)

        aug_bboxes=dict()
        aug_results=dict()
        for key in aug_dict:
            aug_img, aug_bbox = aug_dict[key](image,bboxes)
        
            aug_result = self.predict(aug_img)
            aug_bboxes[key]=aug_bbox
            aug_results[key]=aug_result
        
        return aug_bboxes, aug_results

def _ious(boxes1, boxes2):
    """
    args:
        boxes1: np.array, (N, 4)
        boxes2: np.array, (M, 4)
    return:
        iou: np.array, (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    iner_area = intersect(boxes1, boxes2)
    area1 = area1.reshape(-1, 1).repeat(area2.shape[0], axis=1)
    area2 = area2.reshape(1, -1).repeat(area1.shape[0], axis=0)
    iou = iner_area / (area1 + area2 - iner_area + 1e-14)
    return iou

if __name__ == "__main__":
    miner = MiningCald()
    mining_result = miner.mining()
    rw.write_mining_result(mining_result=mining_result)

