# yolov5

## 环境部署

参考网站：

[【yolov5】三部曲系列教程之GPU环境部署】](https://www.bilibili.com/video/BV1QA411Q7SZ?vd_source=f44e41a06e8bce0da073fdc2f3efc989)

[从零开始完成Yolov5目标识别（一）准备工作](https://blog.csdn.net/WZT725/article/details/123398828)

[AMD显卡不支持CUDA](https://wenku.csdn.net/answer/uez5nn9txn)

安装cuda

[CUDA Toolkit 9.0 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-90-download-archive)

cuda和cuDNN配置

[Cuda和cuDNN安装教程(超级详细)-CSDN博客](https://blog.csdn.net/jhsignal/article/details/111401628)

## kaggle中的python版本

https://blog.csdn.net/Kagiri/article/details/139448198

## 图片标签labelimg

参考网站：  

[目标检测数据集标注工具Labelimg安装与使用](https://blog.csdn.net/qq_45368632/article/details/131810860)

### labelimg打框闪退

参考网站：

[labelimg 打框就闪退 TypeError: setValue(self, int): argument 1 has unexpected type ‘float‘_in scrollrequest bar.setvalue(bar.value() + bar.si-CSDN博客](https://blog.csdn.net/m0_74232237/article/details/130985914)

[【解决labelimg标注图片闪退问题】drawLine(self, l: QLineF): argument 1 has unexpected type ‘float‘_typeerror: arguments did not match any overloaded -CSDN博客](https://blog.csdn.net/kagcee/article/details/135723674)

总结：  

> 1.下载python3.9  
> 2.在pycharm中新建一个python3.9的虚拟环境  
> 3.在pycharm虚拟环境中进行labelimg的运行

## 如何选取数据训练指标

参考网站：
[深度学习图像分类常见问题以及训练技巧 - 知乎 (zhihu.com)深度学习图像分类常见问题以及训练技巧 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/270018811)

## yolov5命令

## Yolov5入门

[半小时搞定Yolov5安装配置及使用（详细过程）-CSDN博客](https://blog.csdn.net/HowieXue/article/details/118445766)  
yolov5 github网址：
[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)  
yolov5 训练模型下载网址：
[Releases · ultralytics/yolov5 (github.com)](https://github.com/ultralytics/yolov5/releases)  

1.图片测试  
--weights 指定模型有问题 指定其他路径下的模型会从github上进行下载（已解决）  
--weights 会从根目录下寻找模型  
下载的模型可集中放在weights文件夹下进行管理  
使用./weights/  
也可以是一个目录

`python detect.py --source ./data/images/bus.jpg --weights ./weights/yolov5s.pt`

2.摄像头测试  
使用本机摄像头进行测试 关闭不了程序和摄像头(已解决)  
按q退出  

`python detect.py --source 0 --weights ./weights/yolov5l.pt`

## Yolov5数据集

参考网站：  
[YOLO目标检测数据集大全【含voc(xml)、coco(json)和yolo(txt)三种格式标签+划分脚本+训练教程】（持续更新建议收藏）](https://blog.csdn.net/m0_64879847/article/details/132301975)

## Yolov5训练自定义数据集

参考网站：

[yolov5训练自定义数据集指南](https://docs.ultralytics.com/zh/yolov5/tutorials/train_custom_data/#13-prepare-dataset-for-yolov5)

[YOLOv5训练自己的数据集(超详细)-CSDN博客](https://blog.csdn.net/qq_40716944/article/details/118188085)

[Yolov5训练自己的数据集（详细完整版）_yolov5缔宇-CSDN博客](https://blog.csdn.net/qq_45945548/article/details/121701492)

[YOLOv5训练自己的数据集(超详细)-CSDN博客](https://blog.csdn.net/qq_40716944/article/details/118188085)

[Yolov5训练自己的数据集（详细完整版）_yolov5缔宇-CSDN博客](https://blog.csdn.net/qq_45945548/article/details/121701492)