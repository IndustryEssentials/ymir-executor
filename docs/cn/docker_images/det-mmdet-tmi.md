# mmdetection 镜像说明文档

## 仓库地址

> 参考[mmdetection](https://github.com/open-mmlab/mmdetection)

- [det-mmdetection-tmi](https://github.com/modelai/ymir-executor-fork/det-mmdetection-tmi)

## 镜像地址
```
youdaoyzbx/ymir-executor:ymir2.0.0-mmdet-cu111-tmi
youdaoyzbx/ymir-executor:ymir2.0.2-mmdet-cu111-tmi
```

## 性能表现

> 参考[mmdetection官方数据](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/README.md)

|  Backbone  | size | Mem (GB) | box AP |                                                  Config                                                   |                                                                                                                                         Download                                                                                                                                         |
| :--------: | :--: | :------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOX-tiny | 416  |   3.5    |  32.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_tiny_8x8_300e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234.log.json) |
|  YOLOX-s   | 640  |   7.6    |  40.5  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_s_8x8_300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711.log.json)       |
|  YOLOX-l   | 640  |   19.9   |  49.4  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_l_8x8_300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236.log.json)       |
|  YOLOX-x   | 640  |   28.1   |  50.9  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_x_8x8_300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254.log.json)       |

**说明**:

1. The test score threshold is 0.001, and the box AP indicates the best AP.
2. Due to the need for pre-training weights, we cannot reproduce the performance of the `yolox-nano` model. Please refer to https://github.com/Megvii-BaseDetection/YOLOX/issues/674 for more information.
3. We also trained the model by the official release of YOLOX based on [Megvii-BaseDetection/YOLOX#735](https://github.com/Megvii-BaseDetection/YOLOX/issues/735) with commit ID [38c633](https://github.com/Megvii-BaseDetection/YOLOX/tree/38c633bf176462ee42b110c70e4ffe17b5753208). We found that the best AP of `YOLOX-tiny`, `YOLOX-s`, `YOLOX-l`, and `YOLOX-x` is 31.8, 40.3, 49.2, and 50.9, respectively. The performance is consistent with that of our re-implementation (see Table above) but still has a gap (0.3~0.8 AP) in comparison with the reported performance in their [README](https://github.com/Megvii-BaseDetection/YOLOX/blob/38c633bf176462ee42b110c70e4ffe17b5753208/README.md#benchmark).


## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| config_file |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| config_file | configs/yolox/yolox_tiny_8x8_300e_coco.py | 文件路径 | mmdetection配置文件 | 建议采用yolox系列, 参考[det-mmdetection-tmi/configs](https://github.com/modelai/ymir-executor-fork/tree/master/det-mmdetection-tmi/configs) |
| samples_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| max_epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| args_options | '' | 字符串 | 训练命令行参数 | 参考 [det-mmdetection-tmi/tools/train.py](https://github.com/modelai/ymir-executor-fork/blob/master/det-mmdetection-tmi/tools/train.py)
| cfg_options | '' | 字符串 | 训练命令行参数 | 参考 [det-mmdetection-tmi/tools/train.py](https://github.com/modelai/ymir-executor-fork/blob/master/det-mmdetection-tmi/tools/train.py)
| metric | bbox | 字符串 | 模型评测方式 | 采用默认值即可 |
| val_interval | 1 | 整数 | 模型在验证集上评测的周期 | 设置为1，每个epoch可评测一次 |
| max_keep_checkpoints | 1 | 整数 | 最多保存的权重文件数量 | 设置为k, 可保存k个最优权重和k个最新的权重文件，设置为-1可保存所有权重文件。

**说明**
1. config_file 可查看[det-mmdetection-tmi/configs](https://github.com/modelai/ymir-executor-fork/tree/master/det-mmdetection-tmi/configs)进行选择


## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| conf_threshold | 0.2 | 浮点数 | 推理结果置信度过滤阈值 | 设置为0可保存所有结果，设置为0.6可过滤大量结果 |
| cfg_options | '' | 字符串 | 训练命令行参数 | 参考 [det-mmdetection-tmi/tools/train.py](https://github.com/modelai/ymir-executor-fork/blob/master/det-mmdetection-tmi/tools/train.py)

**说明**
1. 由于没有采用批量推理技术，因此没有samples_per_gpu和workers_per_gpu选项


## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| mining_algorithm | aldd | 字符串 | 挖掘算法可选 aldd, cald, entropy 和 random | 单类建议采用aldd, 多类检测建议采用entropy |
| cfg_options | '' | 字符串 | 训练命令行参数 | 参考 [det-mmdetection-tmi/tools/train.py](https://github.com/modelai/ymir-executor-fork/blob/master/det-mmdetection-tmi/tools/train.py)

**说明**
1. class_distribution_scores 一些复杂的参数在此不做说明
2. 由于没有采用批量推理技术，因此没有samples_per_gpu和workers_per_gpu选项

## 论文引用

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
