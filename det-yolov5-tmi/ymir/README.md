# yolov5-ymir readme
update 2022/11/23

## build your ymir-executor

```
docker build -t your/ymir-executor:ymir2.0.0-cuda102-yolov5-tmi -f cuda102.dockerfile .

docker build -t your/ymir-executor:ymir2.0.0-cuda111-yolov5-tmi -f cuda111.dockerfile .

docker build -t your/ymir-executor:ymir2.0.0-yolov5-cpu-tmi -f cpu.dockerfile .
```

## 训练: training

### 性能表现

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|[YOLOv5n]      |640  |28.0   |45.7   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|[YOLOv5s]      |640  |37.4   |56.8   |98     |6.4    |0.9    |7.2    |16.5
|[YOLOv5m]      |640  |45.4   |64.1   |224    |8.2    |1.7    |21.2   |49.0
|[YOLOv5l]      |640  |49.0   |67.3   |430    |10.1   |2.7    |46.5   |109.1
|[YOLOv5x]      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|[YOLOv5n6]     |1280 |36.0   |54.4   |153    |8.1    |2.1    |3.2    |4.6
|[YOLOv5s6]     |1280 |44.8   |63.7   |385    |8.2    |3.6    |16.8   |12.6
|[YOLOv5m6]     |1280 |51.3   |69.3   |887    |11.1   |6.8    |35.7   |50.0
|[YOLOv5l6]     |1280 |53.7   |71.3   |1784   |15.8   |10.5   |76.8   |111.4

### 训练参数说明

- 一些参数由ymir后台生成，如 `gpu_id`, `class_names` 等参数
  - `gpu_id`: 使用的GPU硬件编号，如`0,1,2`，类型为 `str`。实际上对应的主机GPU随机，可能为`3,1,7`，镜像中只能感知并使用`0,1,2`作为设备ID。
  - `task_id`: ymir任务id, 类型为 `str`
  - `pretrained_model_params`: 预训练模型文件的路径，类型为 `List[str]`
  - `class_names`: 类别名，类型为 `List[str]`

- 一些参数由ymir后台进行处理，如 `shm_size`, `export_format`， 其中 `shm_size` 影响到docker镜像所能使用的共享内存，若过小会导致 `out of memory` 等错误。 `export_format` 会决定docker镜像中所看到数据的格式



| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| model | yolov5s | 字符串 | yolov5模型，可选yolov5n, yolov5s, yolov5m, yolov5l等 | 建议：速度快选yolov5n, 精度高选yolov5l, yolov5x, 平衡选yolov5s或yolov5m |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| epochs | 100 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| img_size | 640 | 整数 | 输入模型的图像分辨率 | - |
| opset | 11 | 整数 | onnx 导出参数 opset | 建议：一般不需要用到onnx，不必改 |
| args_options | '--exist-ok' | 字符串 | yolov5命令行参数 | 建议：专业用户可用yolov5所有命令行参数 |
| save_best_only | True | 布尔型 | 是否只保存最优模型 | 建议：为节省空间设为True即可 |
| save_period | 10 | 整数 | 保存模型的间隔 | 建议：当save_best_only为False时，可保存 `epoch/save_period` 个中间结果
| sync_bn | False | 布尔型 | 是否同步各gpu上的归一化层 | 建议：开启以提高训练稳定性及精度 |
| activate | '' | 字符串 | 激活函数，默认为nn.Hardswish(), 参考 [pytorch激活函数](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) | 可选值: ELU, Hardswish, LeakyReLU, PReLU, ReLU, ReLU6, SiLU, ... |
| ymir_saved_file_patterns | '' | 字符串 | 用 `,` 分隔的保存文件模式 | 建议：专业用户当希望过滤保存的文件以节省空间时，可设置配置的正则表达式 |

### 训练结果文件示例
```
.
├── data.yaml  # ymir数据集转换后生成的data.yaml
├── models # 模型保存目录
├── monitor.txt  # ymir进度接口文件
├── tensorboard  # tensorboard日志文件
│   ├── events.out.tfevents.1669112949.2cf0844ff367.337.0
│   ├── results.csv
│   ├── results.png
│   ├── train_batch0.jpg
│   ├── train_batch1.jpg
│   └── train_batch2.jpg
├── test.tsv  # ymir数据集转换后生成的测试索引文件，为空
├── train.cache  # 训练集缓存文件
├── train.tsv  # ymir数据集转换后生成的训练集索引文件
├── val.cache  # 验证集缓存文件
└── val.tsv  # ymir数据集转换后生成的测试集索引文件
```


---

## 推理: infer

推理任务中，ymir后台会生成参数 `gpu_id`, `class_names`, `task_id` 与 `model_param_path`， 其中`model_param_path`与训练任务中的`pretrained_model_params`类似。

### 推理参数说明
| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| img_size | 640 | 整数 | 模型的输入图像大小 | 采用32的整数倍，224 = 32*7 以上大小 |
| conf_thres | 0.25 | 浮点数 | 置信度阈值 | 采用默认值 |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值 | 采用默认值 |
| batch_size_per_gpu | 16 | 整数| 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加1倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数| 每张GPU对应的数据读取进程数 | - |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| pin_memory | False | 布尔型 | 是否为数据集单独固定内存? | 内存充足时改为True可加快数据集加载 |


---

## 挖掘: mining

挖掘任务中，ymir后台会生成参数 `gpu_id`, `class_names`, `task_id` 与 `model_param_path`， 其中`model_param_path`与训练任务中的`pretrained_model_params`类似。推理与挖掘任务ymir后台生成的参数一样。

### 挖掘参数说明

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| img_size | 640 | 整数 | 模型的输入图像大小 | 采用32的整数倍，224 = 32*7 以上大小 |
| mining_algorithm | aldd | 字符串 | 挖掘算法名称，可选 random, aldd, cald, entropy | 建议单类检测采用aldd，多类检测采用entropy |
| class_distribution_scores | '' | List[float]的字符表示 | aldd算法的类别平衡参数 | 不用更改， 专业用户可根据各类比较调整，如对于4类检测，用 `1.0,1.0,0.1,0.2` 降低后两类的挖掘比重 |
| conf_thres | 0.25 | 浮点数 | 置信度阈值 | 采用默认值 |
| iou_thres | 0.45 | 浮点数 | nms时的iou阈值 | 采用默认值 |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加1倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| shm_size | 128G | 字符串 | 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| pin_memory | False | 布尔型 | 是否为数据集单独固定内存? | 内存充足时改为True可加快数据集加载 |

## 主要改动：main change log

- add `start.py` and `ymir/ymir_yolov5.py` for train/infer/mining

- add `ymir/ymir_yolov5.py` for useful functions

    - `get_merged_config()` add ymir path config `cfg.yaml` and hyper-parameter `cfg.param`

    - `convert_ymir_to_yolov5()` generate yolov5 dataset config file `data.yaml`

    - `write_ymir_training_result()` save model weight, map and other files.

    - `get_weight_file()` get pretrained weight or init weight file from ymir system

- modify `utils/datasets.py` for ymir dataset format

- modify `train.py` for training process monitor

- add `mining/data_augment.py` and `mining/mining_cald.py` for mining

- add `training/infer/mining-template.yaml` for `/img-man/training/infer/mining-template.yaml`

- add `cuda102/111.dockerfile`, remove origin `Dockerfile`

- modify `requirements.txt`

- other modify support onnx export, not important.

## 更新功能：new features

- 2022/09/08: add aldd active learning algorithm for mining task. [Active Learning for Deep Detection Neural Networks (ICCV 2019)](https://gitlab.com/haghdam/deep_active_learning)
- 2022/09/14: support change hyper-parameter `num_workers_per_gpu`
- 2022/09/16: support change activation, view [rknn](https://github.com/airockchip/rknn_model_zoo/tree/main/models/vision/object_detection/yolov5-pytorch)
- 2022/10/09: fix dist.destroy_process_group() hang
