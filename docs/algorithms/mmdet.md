# ymir-mmdetection

此文档采用 `mmdetection v3.x` 架构，阅读此文档前，建议先了解[mmengine](https://mmengine.readthedocs.io/zh_CN/latest/get_started/introduction.html).

- [mmdetection v3.x](https://github.com/open-mmlab/mmdetection/tree/3.x)

- [ymir-mmdetection](https://github.com/modelai/ymir-mmdetection)

## mmdetection --> ymir-mmdetection

- mmdetection支持 `coco` 与 `pascal voc` 等多种数据格式。ymir-mmdetection镜像会将ymir平台的检测数据格式 `det-ark:raw` 转换为 `coco`。

- mmdetection通过配置文件如 [configs/_base_/datasets/coco_detection.py](https://github.com/open-mmlab/mmdetection/blob/3.x/configs/_base_/datasets/coco_detection.py#L36-L42) 指明数据集的路径。

```
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
    )
```

- 为加载ymir平台数据集，一种方案是参考[自定义数据集](https://mmdetection.readthedocs.io/en/3.x/user_guides/train.html#train-with-customized-datasets)，提供配置文件。但这种方案会固定数据集的类别，不适合ymir平台。

- ymir-mmdetection采用另一种方案，在已有配置文件的基础上，直接在内存中进行修改。参考[ymir-mmyolo/ymir/tools/train.py](https://github.com/modelai/ymir-mmyolo/blob/ymir/tools/train.py#L65-L67)

```
    # 加载已有配置文件如 `configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py`
    cfg = Config.fromfile(args.config)
    # 获得ymir平台超参数
    ymir_cfg = get_merged_config()
    # 直接在内存中修改配置
    modify_mmengine_config(cfg, ymir_cfg)
```

## 配置镜像环境

## 提供超参数模板文件与镜像配置文件

- [img-man/*-template.yaml](https://github.com/modelai/ymir-mmdetection/tree/ymir/ymir/img-man)

## 提供默认启动脚本

- [ymir/start.py](https://github.com/modelai/ymir-mmyolo/tree/ymir/ymir/start.py)

- Dockerfile
```
RUN echo "python /app/ymir/start.py" > /usr/bin/start.sh  # 生成启动脚本 /usr/bin/start.sh
CMD bash /usr/bin/start.sh  # 将镜像的默认启动脚本设置为 /usr/bin/start.sh
```

## 实现基本功能

### 训练

### 推理

### 挖掘

## 制作镜像 det/mmdet:tmi

- [ymir/Dockerfile](https://github.com/modelai/ymir-mmdetection/tree/ymir/ymir/Dockerfile)

```
docker build -t det/mmdet:tmi -f ymir/Dockerfile .
```
