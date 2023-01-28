# ymir2.0.0 (2022-09-30)

- 支持分开输出模型权重，用户可以采用epoch10.pth进行推理，也可以选择epoch20.pth进行推理

- 训练镜像需要指定数据集标注格式, ymir1.1.0默认标注格式为`ark:raw`, ymir2.0.0默认标注格式为`ark:voc`

- 训练镜像可以获得系统的ymir接口版本，方便镜像兼容

- 预训练模型文件在ymir1.1.0时放在/in/models目录下，ymir2.0.0时放在 /in/models/<stage_name>目录下

## 辅助库

- [ymir-executor-sdk](https://github.com/modelai/ymir-executor-sdk) 采用ymir2.0.0分支

- [ymir-executor-verifier](https://github.com/modelai/ymir-executor-verifier) 镜像检查工具

# ymir1.1.0

- [custom ymir-executor](https://github.com/IndustryEssentials/ymir/blob/dev/dev_docs/ymir-dataset-zh-CN.md)

- [ymir-executor-sdk](https://github.com/modelai/ymir-executor-sdk) 采用ymir1.0.0分支
