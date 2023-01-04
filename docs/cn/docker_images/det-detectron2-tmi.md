# detectron2 镜像说明文档

## 代码仓库

> 参考[facebook/detectron2](https://github.com/facebookresearch/detectron2)
- [modelai/ymir-detectron2](https://github.com/modelai/ymir-detectron2)

## 镜像地址
```
youdaoyzbx/ymir-exectutor:ymir2.0.0-detectron2-cu111-tmi
```

## 性能表现

> 数据参考[detectron2/Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50 R50 R101 --fields lr_sched train_speed inference_speed mem box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml">R50</a></td>
<td align="center">1x</td>
<td align="center">0.205</td>
<td align="center">0.041</td>
<td align="center">4.1</td>
<td align="center">37.4</td>
<td align="center">190397773</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml">R50</a></td>
<td align="center">3x</td>
<td align="center">0.205</td>
<td align="center">0.041</td>
<td align="center">4.1</td>
<td align="center">38.7</td>
<td align="center">190397829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml">R101</a></td>
<td align="center">3x</td>
<td align="center">0.291</td>
<td align="center">0.054</td>
<td align="center">5.2</td>
<td align="center">40.4</td>
<td align="center">190397697</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/metrics.json">metrics</a></td>
</tr>
</tbody></table>


## 训练参数

| 超参数 | 默认值 | 类型 | 说明 | 建议 |
| - | - | - | - | - |
| hyper-parameter | default value | type | note | advice |
| batch_size | 2 | 整数 | batch size 大小 | - |
| config_file | configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml | 文件路径 | 配置文件路径 | 参考 [configs/COCO-Detection](https://github.com/modelai/ymir-detectron2/tree/ymir/configs/COCO-Detection) |
| max_iter | 90000 | 整数 | 最大训练次数 | - |
| learning_rate | 0.001 | 浮点数 | 学习率 | - |
| args_options | '' | 字符串 | 命令行参数 | 参考 [default_argument_parser](https://github.com/modelai/ymir-detectron2/blob/ymir/detectron2/engine/defaults.py) |

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
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
