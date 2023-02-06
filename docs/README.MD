# ymir镜像文档

## 简介

- [ymir简介](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E)

- ymir镜像: 为ymir提供模型训练，推理，挖掘服务的镜像

## 快速使用

- [安装ymir](https://github.com/IndustryEssentials/ymir/blob/master/README_zh-CN.md#2-%E5%AE%89%E8%A3%85)

- [ymir操作说明](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E)

- [镜像社区](http://pubimg.vesionbook.com:8110/img) 下载、发布ymir镜像

- 💫[历史镜像](./official-docker-image.md)

- 💫[挖掘算法评测](./mining-images-overview.md)

## Ymir镜像说明

- [Ymir镜像整体流程](./overview/framework.md)

- [Ymir镜像数据集格式](./overview/dataset-format.md)

- [Ymir镜像超参数](./overview/hyper-parameter.md)

- [Ymir镜像制作简介](./overview/ymir-executor.md)

## 基于已有镜像进行定制

- [增/删/改: 默认超参数](./hyper-parameter.md)

## 基于开源仓库进行定制

- [yolov5示例](https://github.com/modelai/ymir-yolov5/pull/2/files)

## 💫 生态环境

- [ymir镜像开发SDK](https://github.com/modelai/ymir-executor-sdk)

    - [读取配置与数据](https://github.com/modelai/ymir-executor-sdk/blob/master/docs/read.md)

    - [写进度与结果文件](https://github.com/modelai/ymir-executor-sdk/blob/master/docs/write.md)

    - [数据集格式转换](https://github.com/modelai/ymir-executor-sdk/blob/master/docs/dataset_convert.md)

- [ymir镜像调试校验工具](https://github.com/modelai/ymir-executor-verifier)

    - 样例数据下载

    - 交互式调试

    - 批量校验镜像

## FAQ

- [FAQ](./FAQ.md)

## 其它

- [ymir版本与接口兼容](./ymir-executor-version.md)

- [加速apt/pip/docker](./speedup_apt_pip_docker.md)

- [导入外部模型权值](./import_outer_weight.md)
