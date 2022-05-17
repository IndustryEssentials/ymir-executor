"""
convert ymir dataset to yolov5 dataset
"""
import shutil
import os
import os.path as osp
import imagesize
import yaml
import numpy as np
from executor import dataset_reader as dr, env, monitor, result_writer as rw
from utils.datasets import img2label_paths

def convert_ymir_to_yolov5(root_dir):
    os.makedirs(root_dir,exist_ok=True)
    os.makedirs(osp.join(root_dir,'images'),exist_ok=True)
    os.makedirs(osp.join(root_dir,'labels'),exist_ok=True)

    train_data_size=dr.dataset_size(env.DatasetType.TRAINING)
    val_data_size=dr.dataset_size(env.DatasetType.VALIDATION)
    N=len(str(train_data_size+val_data_size))
    idx=0
    
    DatasetTypeDict=dict(train=env.DatasetType.TRAINING,
        val=env.DatasetType.VALIDATION)

    path_env = env.get_current_env()
    for split in ['train','val']:
        split_imgs=[]
        for asset_path, annotation_path in dr.item_paths(dataset_type=DatasetTypeDict[split]):
            idx+=1
            asset_path=osp.join(path_env.input.root_dir, path_env.input.assets_dir, asset_path)
            annotation_path=osp.join(path_env.input.root_dir, path_env.input.annotations_dir, annotation_path)
            assert osp.exists(asset_path),f'cannot find {asset_path}'
            assert osp.exists(annotation_path),f'cannot find {annotation_path}'

            img_suffix=osp.splitext(asset_path)[1]
            img_path=osp.join(root_dir,'images',str(idx).zfill(N)+img_suffix)
            ann_path=osp.join(root_dir,'labels',str(idx).zfill(N)+'.txt')
            yolov5_ann_path=img2label_paths([img_path])[0]
            assert  yolov5_ann_path== ann_path, f'bad yolov5_ann_path={yolov5_ann_path} and ann_path = {ann_path}'

            shutil.copy(asset_path, img_path)
            width, height = imagesize.get(img_path)
            
            with open(ann_path,'w') as fw:
                with open(annotation_path,'r') as fr:
                    for line in fr.readlines():
                        class_id, xmin, ymin, xmax, ymax = [int(x) for x in line.strip().split(',')]

                        # class x_center y_center width height
                        # normalized xywh
                        # class_id 0-indexed
                        xc=(xmin+xmax)/2/width 
                        yc=(ymin+ymax)/2/height 
                        w=(xmax-xmin)/width 
                        h=(ymax-ymin)/height
                        fw.write(f'{class_id} {xc} {yc} {w} {h}\n')

            split_imgs.append(img_path)
        with open(osp.join(root_dir,f'{split}.txt'),'w') as fw:
            fw.write('\n'.join(split_imgs))

    ### generate yaml
    config=env.get_executor_config()
    data=dict(path=root_dir,
        train="train.txt",
        val="val.txt",
        nc=len(config['class_names']),
        names=config['class_names'])

    with open('data.yaml','w') as fw:
        fw.write(yaml.safe_dump(data))

def write_ymir_training_result(results, maps, rewrite=False):
    """
    results: (mp, mr, map50, map, loss)
    """
    if not rewrite:
        training_result_file = env.get_current_env().output.training_result_file
        if osp.exists(training_result_file):
            return 0

    model = env.get_executor_config()['model']
    class_names = env.get_executor_config()['class_names']
    map50=maps

    #! use `rw.write_training_result` to save training result
    rw.write_training_result(model_names=[f'{model}.yaml', 'best.pt', 'last.pt', 'best.onnx'],
                             mAP=float(np.mean(map50)),
                             classAPs={class_name: v
                                       for class_name, v in zip(class_names,map50.tolist())})
    return 0


if __name__ == '__main__':
    convert_ymir_to_yolov5('/out/yolov5_dataset')
    

