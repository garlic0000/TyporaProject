# Yolov5入门

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
