## active_learning镜像说明

### 1.功能描述

使用各类active_learning方法，配合实际业务策略，对图片打分，筛选，起到挑选数据的作用

**启动方式如下**

```shell
nvidia-docker run -it --rm -v /data/in:/in -v /data/out:/out al:0.0.1
```

**挂载点约定如下**

**/in/candidate**
含有一个索引文件index.tsv，里面每一行的内容为：

<assert_path> (相对路径，相对于index.tsv)

**/in/model**
推理使用的模型，对mxnet而言，模型包括两个文件，命名规则如下所示

xxx-epoch.params

xxx-symbol.json

mobilenet_sc_cpu_combined_aldd_select_iter2_2000-0130.params

mobilenet_sc_cpu_combined_aldd_select_iter2_2000-symbol.json

**/in/config.yaml**
运行时的配置文件

**/out/log.txt**
非必要log

**/out/result.tsv**
输出结果

**/out/monitor.txt**
必要，进度log，可附加异常结束的错误信息

**/out/monitor-log.txt**
必要，状态切换log

**docker内文件说明**

/app 代码位置

/img-man/readme.md 本文件

/img-man/config-template.yaml 配置文件模板

### 2.输出结果样式

所有log均以"\t"作为间隔符号，在/out/result.tsv中，输出内容为

assert_path score

可以有多个score输出，例如

assert_path score1 score2

**score均为越高越好，数值范围受到具体al算法影响，不固定，输出顺序按照score1从高到低排列**