# 测试Ymir目标检测镜像

## 通过YMIR平台进行测试

用户可以直接通过Ymir平台发起目标检测的训练，推理及挖掘任务，对镜像进行测试。、

!!! 注意
    YMIR平台发起的任务在顺利结束时，会清理相应的目录，因此在测试时，请确保相应目录存在。

### 导入待测镜像

- 假设用户已经制作好镜像 **demo/det:tmi**, 它支持训练、推理及挖掘

- 假设用户具有管理员权限，按照[新增镜像](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E#%E6%96%B0%E5%A2%9E%E9%95%9C%E5%83%8F) 将**demo/det:tmi** 添加到 **我的镜像** 中。

### 导入待测数据集

- 下载示例目标检测数据集 [ymir2.0.0_dog_train.zip](https://github.com/modelai/ymir-executor-fork/releases/download/dataset-ymir2.0.0/ymir2.0.0_dog_train.zip)  [ymir2.0.0_dog_val.zip](https://github.com/modelai/ymir-executor-fork/releases/download/dataset-ymir2.0.0/ymir2.0.0_dog_val.zip)

- 建立包含对应标签的项目， `训练类别` 中添加对应标签 `dog`

- 按照[添加数据集](https://github.com/IndustryEssentials/ymir/wiki/%E6%93%8D%E4%BD%9C%E8%AF%B4%E6%98%8E#%E6%B7%BB%E5%8A%A0%E6%95%B0%E6%8D%AE%E9%9B%86)导入示例目标检测数据集

### 发起待测任务

发起待测的训练、推理或挖掘任务后，等待其结束或出错。

### 获取任务id

登录服务器后台，进入YMIR部署的目录 `ymir-workplace`

- 对于训练任务：`cd sandbox/work_dir/TaskTypeTraining`

- 对于挖掘或推理任务： `cd sandbox/work_dir/TaskTypeMining`

- 对于单张图片测试任务： `cd sandbox/work_dir/TaskTypeInfer`

- 列举当前所有的任务，按任务时间找到对应任务id, 此处假设为最新的 **t00000020000023a473e1673591617**

!!! 注意
    对于训练任务， 可通过tensorboard链接获得对应任务id。

```
> ls -lt .

drwxr-xr-x 4 root root 45 Jan 13 14:33 t00000020000023a473e1673591617
drwxr-xr-x 4 root root 45 Jan 13 14:19 t00000020000025d55ff1673590756
drwxr-xr-x 4 root root 45 Jan 13 14:13 t00000020000028b0cce1673590425
drwxr-xr-x 4 root root 45 Jan 10 14:09 t00000020000018429301673330944
drwxr-xr-x 4 root root 45 Jan  9 18:21 t000000200000210e0811673259669
drwxr-xr-x 4 root root 45 Jan  9 18:07 t00000020000029e02f61673258829
```

### 通过 docker 进行交互式调试

- 进行任务id对应的工作目录 `cd t00000020000023a473e1673591617/sub_task/t00000020000023a473e1673591617`

- 列举当前目录可以看到 `in` 和 `out` 目录

- 进行交互式调试

    - 假设 `ymir-workplace` 存放在 **/data/ymir/ymir-workplace**, 需要将 `ymir-workplace` 目录也挂载到镜像中相同位置，以确保所有软链接均有效。

    - 假设启动程序为 **/usr/bin/start.sh**

```
# --ipc host 表示容器共享主机的所有内存
docker run -it --rm --gpus all --ipc host -v $PWD/in:/in -v $PWD/out:/out -v /data:/data demo/det:tmi bash

# --shm-size 128g 表示容器最多共享主机128G内存
# docker run -it --rm --gpus all --shm-size 128g -v $PWD/in:/in -v $PWD/out:/out -v /data:/data demo/det:tmi bash

bash /usr/bin/start.sh
```

- 假设用户开发镜像的代码存放在 **/home/modelai/code**， 为方便测试， 可以将 **/home/modelai/code** 也挂载到镜像中进行测试。

    - 假设实际启动程序为 **start.py**

```
docker run -it --rm --gpus all --ipc host -v $PWD/in:/in -v $PWD/out:/out -v /data:/data -v /home/modelai/code:/home/modelai/code demo/det:tmi bash

cd /home/modelai/code
python start.py
```

### 测试通过后

- 通过 `docker build` 重新构建镜像， 如果修改了超参数，需要在Ymir平台删除旧镜像并重新添加，使更新的超参数生效。如果仅仅修改了代码，不需要重新添加即可使用本地的最新镜像。


## 💫 YMIR后台错误查看

- 如镜像正确运行，但输出格式不符合YMIR后台要求，或其他错误，可在 `ymir-workplace/ymir-data/logs` 下查看

```
tail -f -n 200 ymir_controller.log
```

## 💫 通过 ymir-executor-verifier 进行测试

[ymir-executor-verifier](https://github.com/modelai/ymir-executor-verifier) 面向企业用户，目的是对大量镜像进行自动化测试，以保障镜像的质量。
