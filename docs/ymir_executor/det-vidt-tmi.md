# vidt 镜像说明文档

ICLR 2022的 transformer 架构检测器

## 代码仓库

> 参考[naver-ai/vidt](https://github.com/naver-ai/vidt)
- [modelai/ymir-vidt](https://github.com/modelai/ymir-vidt)

## 镜像地址
```
youdaoyzbx/ymir-executor:ymir2.0.0-vidt-cu111-tmi
```

## 性能表现

> 数据参考[naver-ai/vidt](https://github.com/naver-ai/vidt)

| Backbone | Epochs | AP | AP50 | AP75 | AP_S | AP_M | AP_L | Params | FPS | Checkpoint / Log |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| `Swin-nano` | 50 (150) | 40.4 (42.6) | 59.9 (62.2) | 43.0 (45.7) | 23.1 (24.9) | 42.8 (45.4) | 55.9 (59.1) | 16M | 20.0 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_nano_150.txt))|
| `Swin-tiny` | 50 (150)| 44.9 (47.2) | 64.7 (66.7) | 48.3 (51.4) | 27.5 (28.4) | 47.9 (50.2) | 61.9 (64.7) | 38M | 17.2 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_tiny_150.txt))|
| `Swin-small` | 50 (150) | 47.4 (48.8) | 67.7 (68.8) | 51.2 (53.0) | 30.4 (30.7) | 50.7 (52.0) | 64.6 (65.9) | 60M | 12.1 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_small_150.txt))|
| `Swin-base` | 50 (150) | 49.4 (50.4) | 69.6 (70.4) | 53.4 (54.8) | 31.6 (34.1) | 52.4 (54.2) | 66.8 (67.4) | 0.1B | 9.0 | [Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_50.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_50.txt) <br>([Github](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_150.pth) / [Log](https://github.com/naver-ai/vidt/releases/download/v0.1-vidt/vidt_base_150.txt)) |


## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| shm_size | 128G | 字符串| 受ymir后台处理，docker image 可用共享内存 | 建议大小：镜像占用GPU数 * 32G |
| export_format | ark:raw | 字符串| 受ymir后台处理，ymir数据集导出格式 | - |
| backbone_name | swin_nano | 字符串 | 骨架网络，可选swin_nano, swin_tiny, swin_small, swin_base | - |
| batch_size_per_gpu | 16 | 整数 | 每张GPU一次处理的图片数量 | 建议大小：显存占用<50% 可增加2倍加快训练速度 |
| num_workers_per_gpu | 4 | 整数 | 每张GPU对应的数据读取进程数 | - |
| epochs | 50 | 整数 | 整个数据集的训练遍历次数 | 建议：必要时分析tensorboard确定是否有必要改变，一般采用默认值即可 |
| learning_rate | 0.0001 | 浮点数 | 学习率 | - |
| eval_size | 640 | 整数 | 输入网络的图片大小 | - |
| weight_save_interval | 100 | 整数 | 权重文件保存间隔 | - |
| args_options | '' | 字符串 | 命令行参数 | 参考 [get_args_parser](https://github.com/modelai/ymir-vidt/blob/ymir-dev/arguments.py) |

## 推理参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| conf_threshold | 0.2 | 浮点数 | 置信度阈值 | 采用默认值 |

## 挖掘参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| conf_threshold | 0.2 | 浮点数 | 置信度阈值 | 采用默认值 |

## 引用
```
@inproceedings{song2022vidt,
  title={ViDT: An Efficient and Effective Fully Transformer-based Object Detector},
  author={Song, Hwanjun and Sun, Deqing and Chun, Sanghyuk and Jampani, Varun and Han, Dongyoon and Heo, Byeongho and Kim, Wonjae and Yang, Ming-Hsuan},
  booktitle={International Conference on Learning Representation},
  year={2022}
}
```
