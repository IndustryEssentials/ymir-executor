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

## 💫复杂用法

!!! 注意
    这部分内容初学者可以跳过

### cfg_options

当用户使用脚本 “tools/train.py” 或 “tools/test.py” 提交任务，或者其他工具时，可以通过指定 --cfg-options 参数来直接修改配置文件中内容。

- 更新字典链中的配置的键

    配置项可以通过遵循原始配置中键的层次顺序指定。例如，--cfg-options model.backbone.norm_eval=False 改变模型 backbones 中的所有 BN 模块为 train 模式。

- 更新列表中配置的键

    你的配置中的一些配置字典是由列表组成。例如，训练 pipeline data.train.pipeline 通常是一个列表。 例如 [dict(type='LoadImageFromFile'), dict(type='TopDownRandomFlip', flip_prob=0.5), ...]。 如果你想要在 pipeline 中将 'flip_prob=0.5' 修改为 'flip_prob=0.0' ， 您可以指定 --cfg-options data.train.pipeline.1.flip_prob=0.0.

- 更新 list/tuples 中的值

    如果想要更新的值是一个列表或者元组。 例如, 一些配置文件中包含 param_scheduler = "[dict(type='CosineAnnealingLR',T_max=200,by_epoch=True,begin=0,end=200)]"。 如果你想要改变这个键，你可以指定 --cfg-options param_scheduler = "[dict(type='LinearLR',start_factor=1e-4, by_epoch=True,begin=0,end=40,convert_to_iter_based=True)]"。 注意, ” 是必要的, 并且在指定值的时候，在引号中不能存在空白字符。


