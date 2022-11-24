# docker 加速 apt

在 `dockerfile` 中添加如下命令再进行 `apt` 安装

```
# Install linux package
RUN	sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && apt-get update
```

- [ubuntu/debian 加速apt](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)
- [centos 加速yum](https://mirrors.tuna.tsinghua.edu.cn/help/centos/)

# docker 加速 pip

在 `dockerfile` 中添加如下命令再进行 `pip` 安装

```
# install ymir-exc sdk
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

- [pip 加速](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
- [conda/anaconda 加速](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)


# docker pull/push 加速

以下链接均没测试，欢迎反馈

- [南京大学 mirror](https://nju-mirror-help.njuer.org/dockerhub.html)

- [百度网易阿里 mirror](https://yeasy.gitbook.io/docker_practice/install/mirror)

- [华为 mirror](https://bbs.huaweicloud.com/blogs/381362)
