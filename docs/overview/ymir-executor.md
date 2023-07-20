# ymir镜像制作简介

## 背景知识

- [python3](https://www.runoob.com/python3/python3-tutorial.html) ymir平台，深度学习框架，开源算法库主要以python3进行开发

- [docker](https://www.runoob.com/docker/docker-tutorial.html) 制作ymir镜像，需要了解docker 及 [dockerfile](https://www.runoob.com/docker/docker-dockerfile.html)

- [linux](https://www.runoob.com/linux/linux-shell.html) ymir镜像主要基于linux系统，需要了解linux 及 [linux-shell](https://www.runoob.com/linux/linux-shell.html)

- [深度视觉算法] ymir镜像的核心算法是深度视觉算法，需要了解[深度学习](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning)， 计算机视觉。

- [深度学习框架] 应用深度学习算法离不开深度学习框架如 [pytorch](https://pytorch.org/), [tensorflow](https://tensorflow.google.cn/?hl=en) 与 [keras](https://keras.io/) 等的支持。熟悉其中的一种即可，推荐pytorch.

- [深度学习算法库] 基于已有的算法库应用前沿算法或开发新算法是常规操作，推荐了解 [mmdetection](https://github.com/open-mmlab/mmdetection) 与 [yolov5](https://github.com/ultralytics/yolov5)


## 环境依赖

假设拥有一台带nvidia显卡的linux服务器, 以ubuntu18.04 为例

!!! 注意
    如果apt update 或 apt install 速度缓慢，可以考虑更换软件源
    [清华软件源](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)
    [中科大软件源](http://mirrors.ustc.edu.cn/help/ubuntu.html)

- [docker](https://www.runoob.com/docker/ubuntu-docker-install.html)
```
# 安装
curl -sSL https://get.daocloud.io/docker | sh

# 测试
sudo docker run hello-world

# 添加普通用户执行权限
sudo usermod -aG docker $USER

# 重新login后测试
docker run hello-world
```

- [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide)

!!! 注意
    先按照上述链接中的前提条件安装好 **NVIDIA Driver >=510.47.03 **, 以支持 `cuda11.6+`

!!! gpu驱动与cuda版本
    引用自openmmlab：

    对于基于 Ampere 的 NVIDIA GPU，例如 GeForce 30 系列和 NVIDIA A100，CUDA 版本需要 >= 11。
    对于较旧的 NVIDIA GPU，CUDA 11 向后兼容，但 CUDA 10.2 提供更好的兼容性并且更轻量级。
    请确保 GPU 驱动程序满足最低版本要求。有关详细信息，请参阅[此表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)

```
# 添加软件源
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 更新索引
sudo apt-get update

# 安装
sudo apt-get install -y nvidia-docker2

# 重启docker
sudo systemctl restart docker

# 测试
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## 制作一个hello world 镜像

### 编辑Dockerfile

```
# vim Dockerfile
# cat Dockerfile

FROM ubuntu:18.04  # 基于ubuntu18.04镜像制作新镜像

CMD echo "hello ymir executor"  # 新镜像在运行时默认执行的命令
```

### 制作 hello-ymir:latest 镜像

```
# docker build -t hello-ymir:latest -f Dockerfile .

Sending build context to Docker daemon  52.74kB
Step 1/2 : FROM ubuntu:18.04
18.04: Pulling from library/ubuntu
a055bf07b5b0: Pull complete
Digest: sha256:c1d0baf2425ecef88a2f0c3543ec43690dc16cc80d3c4e593bb95e4f45390e45
Status: Downloaded newer image for ubuntu:18.04
 ---> e28a50f651f9
Step 2/2 : CMD echo "hello ymir executor"
 ---> Running in 6dd391c7688d
Removing intermediate container 6dd391c7688d
 ---> 4c8672e6ce02
Successfully built 4c8672e6ce02
Successfully tagged hello-ymir:latest
```

### 测试

```
# docker run -it --rm hello-ymir

hello ymir executor
```

## ymir 镜像制作

### 基础镜像

需要选择一个合适的基础镜像来避免从0开始制作ymir镜像，上面的例子中我们采用ubuntu18.04作用基础镜像构建新镜像，基于实践，我们推荐制作ymir镜像的基础镜像包含以下配置：

- python 版本 >= 3.8

- ymir镜像的cuda版本<=主机支持的cuda版本

- 推荐基于[nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags) 与 [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags) 进行ymir镜像制作

### 所有ymir镜像均需要实现的功能

- 提供超参数模板文件： 必选，ymir平台需要解析镜像的 **/img-man** 目录生成超参数配置页面

- 提供默认启动脚本：必选，推荐采用 **bash /usr/bin/start.sh** 作用镜像的默认启动脚本

- 写进度： 必选, 将程序当前完成的百分比反馈到ymir平台，从而估计程序的剩余运行时间

- 写结果文件：必选，将程序运行的结果反馈到ymir平台

- 提供镜像说明文件：可选，ymir平台通过解析 **/img-man/manifest.yaml** 得到镜像的目标类型，即镜像支持目标检测，语义分割还是实例分割。默认目标类型为目标检测。

### 训练镜像需要实现的额外功能

- 基本功能：加载数据集与超参数进行训练，将模型权重，模型精度等结果保存到 **/out** 目录的指定文件。

```
# pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0"
from ymir_exc import env

env_config = env.get_current_env()
with open(env_config.output.training_result_file, "w") as f:
    yaml.safe_dump(data=training_result, stream=f)
```

- 写tensorboard日志：可选, ymir平台支持查看训练任务的tensorboard训练日志

### 推理镜像需要实现的额外功能

- 基本功能：加载数据集与模型权重进行推理，将推理结果保存到 **/out** 目录的指定文件。

```
env_config = env.get_current_env()
with open(env_config.output.infer_result_file, "w") as f:
    f.write(json.dumps(result))
```

### 挖掘镜像需要实现的额外功能

- 基本功能：加载数据集与模型权重进行挖掘，基于主动学习算法获得每张图片的重要程度分数，将分数保存到 **/out** 目录的指定文件。

```
env_config = env.get_current_env()
with open(env_config.output.mining_result_file, "w") as f:
    for asset_id, score in sorted_mining_result:
        f.write(f"{asset_id}\t{score}\n")
```
