# 常见自制镜像错误

## 训练镜像

- 写训练结果精度时数据格式为tensor或numpy等，而不是基本的float
```
result = torch.tensor(0.5)
evaluation_result = dict(mAP=result)  # 应改为 evaluation_result = dict(mAP=result.item())

yaml.representer.RepresenterError: ('cannot represent an object', 0.39)
```
