# ymir 镜像数据标注格式

本文介绍在算法镜像中，ymir的数据标注格式。

| export_format | 算法类型 | 格式说明 |
| - | - | - |
| ark:raw 或 det-ark:raw | 目标检测 | 标注文件为txt |
| voc:raw 或 det-voc:raw | 目标检测 | 标注文件为xml，目标检测默认格式 |
| seg-coco:raw | 图像分割 | 标注文件为json，图像分割默认格式 |

## 设置修改

- 对于训练镜像，用户可以通过设置 `/img-man/training-template.yaml` 中的 `export_format` 字段来控制镜像需要使用数据格式。

- 对于推理或挖掘镜像，由于不需要用到标注文件，因此不需要设置数据标注格式

- 目录结构

```
/in
├── annotations  # 标注文件所在目录
├── assets  # 图像所在目录
├── config.yaml  # 超参数配置文件
├── env.yaml  # ymir环境配置文件
├── models  # 预训练模型权重文件所在目录
├── train-index.tsv  # 训练集索引文件
└── val-index.tsv  # 验证集索引文件
```

- 索引文件格式

每行为 `图像绝对路径` + `\t` + `标注文件绝对路径`，用户按行解析索引文件，即可获得所有的标注图像与标注文件。

## det-ark:raw

- 索引文件示例
```
/in/assets/train/26681097a3e1194777c8dc7bb946e70d0cbbcec8.jpeg  /in/annotations/train/26681097a3e1194777c8dc7bb946e70d0cbbcec8.txt
/in/assets/train/6cf24e81164571c5e5f5f10dc9f51cde13fabb05.jpeg  /in/annotations/train/6cf24e81164571c5e5f5f10dc9f51cde13fabb05.txt
/in/assets/train/07b3fb8bd1e36b5edb509b822c1ad86b5863630f.jpeg  /in/annotations/train/07b3fb8bd1e36b5edb509b822c1ad86b5863630f.txt
```

- 标注文件示例

每行为 `class_id` + `xmin` + `ymin` + `xmax` + `ymax`，通过 `,` 进行分隔。

```
0, 122, 7, 372, 375
1, 211, 147, 325, 255
```

## det-voc:raw

- 索引文件示例
```
/in/assets/train/26681097a3e1194777c8dc7bb946e70d0cbbcec8.jpeg  /in/annotations/train/26681097a3e1194777c8dc7bb946e70d0cbbcec8.xml
/in/assets/train/6cf24e81164571c5e5f5f10dc9f51cde13fabb05.jpeg  /in/annotations/train/6cf24e81164571c5e5f5f10dc9f51cde13fabb05.xml
/in/assets/train/07b3fb8bd1e36b5edb509b822c1ad86b5863630f.jpeg  /in/annotations/train/07b3fb8bd1e36b5edb509b822c1ad86b5863630f.xml
```

- 标注文件示例

参考voc xml 格式

```
<annotation>
	<folder>VOC2012</folder>
	<filename>2008_000026.jpg</filename>
	<source>
		<database>The VOC2008 Database</database>
		<annotation>PASCAL VOC2008</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>person</name>
		<pose>Frontal</pose>
		<truncated>1</truncated>
		<occluded>1</occluded>
		<bndbox>
			<xmin>122</xmin>
			<ymin>7</ymin>
			<xmax>372</xmax>
			<ymax>375</ymax>
		</bndbox>
		<difficult>0</difficult>
	</object>
	<object>
		<name>dog</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<occluded>1</occluded>
		<bndbox>
			<xmin>211</xmin>
			<ymin>147</ymin>
			<xmax>325</xmax>
			<ymax>255</ymax>
		</bndbox>
		<difficult>0</difficult>
	</object>
</annotation>

```

## seg-coco:raw

- 索引文件示例

其中所有图像文件都对应同一个标注文件

```
/in/assets/train/26681097a3e1194777c8dc7bb946e70d0cbbcec8.jpeg  /in/annotations/coco-annotations.json
/in/assets/train/6cf24e81164571c5e5f5f10dc9f51cde13fabb05.jpeg  /in/annotations/coco-annotations.json
/in/assets/train/07b3fb8bd1e36b5edb509b822c1ad86b5863630f.jpeg  /in/annotations/coco-annotations.json
```

- 标注文件示例

参考coco格式

```

```
