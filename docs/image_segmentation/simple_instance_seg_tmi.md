# 制作简单的实例分割镜像

参考语义分割镜像的制作:
- [语义分割-训练](./simple_semantic_seg_training.md)
- [语义分割-推理](./simple_semantic_seg_infer.md)
- [语义分割-挖掘](./simple_semantic_seg_mining.md)

## 镜像说明文件

**object_type** 为 4 表示镜像支持实例分割

- [img-man/manifest.yaml](https://github.com/modelai/ymir-executor-fork/tree/ymir-dev/seg-instance-demo-tmi/img-man/manifest.yaml)
```
# 4 for instance segmentation
"object_type": 4
```

## 训练结果返回

```
rw.write_model_stage(stage_name='epoch20',
                     files=['epoch20.pt', 'config.py'],
                     evaluation_result=dict(maskAP=expected_maskap))
```

## 推理结果返回

采用coco数据集格式，相比语义分割，实例分割的annotation中需要增加 `bbox` 的置信度。
```
# for instance segmentation
annotation_info['confidence'] = min(1.0, 0.1 + random.random())

coco_results = convert(cfg, results, True)
rw.write_infer_result(infer_result=coco_results, algorithm='segmentation')
```
