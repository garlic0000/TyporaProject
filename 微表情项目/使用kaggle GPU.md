# 使用Kaggle GPU

[【YOLOv5】利用Kaggle的GPU训练（运行）yolov5模型（项目）](https://blog.csdn.net/qq_62573714/article/details/137927584)

[解决Kaggele无法下载输出output文件夹下的文件](https://blog.csdn.net/Yslin_/article/details/122353340)

模仿yolov5将代码上传至github，再从kaggle下载github上的代码进行运行，可随时更改和同步代码

## 关于Kaggle Linux

kaggle 服务器的系统为Ubuntu

在2020年注册过一个账号，在2024年7月份中旬左右创建了一个notebook

该notebook的Ubuntu系统版本是Ubuntu 20.04

但是在8月底9月初时，新建notebook的系统版本升级为Ubuntu22.04

由于使用时间不够，又新注册了两个账号，这两个账号新建使用带GPU的notebook时，Ubuntu版本为22.04，但是使用不带GPU版本的notebook时，Ubuntu版本为20.04

总结如下

三个账号

新创建带GPU的notebook时，Ubuntu版本为22.04

不带GPU的notebook，Ubuntu版本为20.04

有一个账号创建的带GPU版本的notebook的较早，只有该notebook为20.04

将该notebook下载后导入其他账号，新的notebook的系统设置会被覆盖为22.04

如要多账号使用同一个notebook的20.04系统，需对该notebook进行共享

设置为私有共享，可以指定共同编辑账号

共同编辑该notebook不会同步改变内容，但是编辑标题会共同改变，且notebook的链接是相同的

## 关于Kaggle notebook

一个账号只能创建5个notebook
