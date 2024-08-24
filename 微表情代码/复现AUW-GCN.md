# AUW-GCN

项目github地址：  
[xjtupanda / AUW-GCN](https://github.com/xjtupanda/AUW-GCN)

## 项目环境配置

> OS: Ubuntu 20.04.4 LTS  
>
> Python: 3.8 
>
> Pytorch: 1.10.1  
>
> CUDA: 10.2, cudnn: 7.6.5  
>
> GPU: NVIDIA GeForce RTX 2080 Ti  

这个环境有点问题
1.配置CUDA10.2，在Ubuntu系统上配置，官网上只提供Ubuntu18.04  

2.这里pytorch为1.10.1，但在requirements.txt中torch版本为1.13.1

先下载CUDA10.7试试

实际实验环境配置：

1.kaggle
误打误撞，环境配置成功过，但是再也没法复现  

    a.python3.8 torch==1.13.1 torchvision==0.11.2+cu102
    报错：需要python3.8以上 匹配不到torchvision==0.11.2+cu102
    
    b.python3.9 torch==1.13.1 torchvision==0.11.2+cu102
    报错：需要python3.9以上 匹配不到torchvision==0.11.2+cu102
    
    c.python3.10 torch==1.13.1 torchvision==0.11.2+cu102
    报错：需要python3.10一下 匹配不到torchvision==0.11.2+cu102
    
    d.python3.8 torch torchvision==0.11.2+cu102
    由于项目配置环境和requirements.txt中torch版本冲突，暂时不指定torch版本，使用torchvision==0.11+cu102版本去匹配
    报错：需要python3.9以上 匹配不到torchvision==0.11.2+cu102
    
    e.python3.8 torch==1.13.1 torchvision
    用torchchvision去匹配torch的版本
    可行 最终下载的torchvision版本为torchvision==0.14.1

2.colab
服务器连接不稳定
没有conda环境

## 复现过程

### 1.配置python环境

### 2.上传项目代码、数据集、模型文件

### 3.数据集处理

### 4.训练和测试模型

### 5.结果下载

[【环境配置篇】保姆级教学之Ubuntu20.04上编译OpenCV+CUDA_ubuntu opencv cuda-CSDN博客](https://blog.csdn.net/ChunjieShan/article/details/125391238)

[最全、最新安装 Denseflow 教程，安装 CUDA11.8、12.4 支持的 OpenCV 4.X【MCPRL】_安装denseflow-CSDN博客](https://blog.csdn.net/baihupleonly/article/details/139360191)

[在kaggle中的notebook 如何自定义 cuda 版本以及如何使用自定义的conda或python版本运行项目（一）_kaggle cuda-CSDN博客](https://blog.csdn.net/Magicapprentice/article/details/139148080)

[Check CUDA and cuDNN (kaggle.com)](https://www.kaggle.com/code/titericz/check-cuda-and-cudnn)

## 运行问题

> This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.

[pytorch中DataLoader的num_workers参数详解与设置大小建议](https://blog.csdn.net/qq_28057379/article/details/115427052)



> UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
>
> 
>
> UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.

[深度学习：UserWarning: The parameter ‘pretrained‘ is deprecated since 0.13..解决办法_userwarning: the parameter 'pretrained' is depreca-CSDN博客](https://blog.csdn.net/qudunan6468/article/details/133808253)

在`feature_extraction/retinaface/models/retinaface.py`中第71行的代码`backbone = models.resnet50(pretrained=cfg['pretrain'])`



> UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired.

[UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1-CSDN博客](https://blog.csdn.net/m0_51233386/article/details/128489132)



> sh: 1: denseflow: not found

[bash: denseflow: command not found_denseflow: not found-CSDN博客](https://blog.csdn.net/qq_45047510/article/details/131333122)

需要安装denseflow

参考网站：

[最全、最新安装 Denseflow 教程，安装 CUDA11.8、12.4 支持的 OpenCV 4.X【MCPRL】_安装denseflow-CSDN博客](https://blog.csdn.net/baihupleonly/article/details/139360191)

[Linux配置Denseflow - Kamino's Blog](https://blog.kamino.link/2022/05/01/Linux配置Denseflow/)

原始的`zzopencv.sh`需要修改，把两个文件的下载链接改为4.5.0(Python3.8.13)



> nasm/yasm not found or too old. Use --disable-x86asm for a crippled build.
>
> If you think configure made a mistake, make sure you are using the latest
> version from Git.  If the latest version fails, report the problem to the
> ffmpeg-user@ffmpeg.org mailing list or IRC #ffmpeg on irc.freenode.net.
> Include the log file "ffbuild/config.log" produced by configure as this will help
> solve the problem.

参考：

[ffmpeg 编译](https://www.cnblogs.com/zhaohu/p/9488805.html)



> /usr/bin/cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)

在`/opt/conda/lib/`路径下有三个这样的文件，删除`libcurl.so.4`

> ```
> libcurl.so
> libcurl.so.4
> libcurl.so.4.8.0
> ```

参考:

[linux cmake error no version information available - HappyCoder_1 - 博客园 (cnblogs.com)](https://www.cnblogs.com/132818Creator/p/13091631.html)



> fatal error: gio/gio.h: No such file or directory



参考：

[fatal error: gio/gio.h: 没有那个文件或目录 - CSDN文库](https://wenku.csdn.net/answer/0d2d12f8f9704a56bfe4cd616b29315b)

[如何安装gio-unix-2.28.0 Ubuntu 中文网 (dovov.com)](https://ubuntu.dovov.com/14536/如何安装gio-unix-2-28-0.html)



> Could NOT find TIFF (missing: TIFF_LIBRARY TIFF_INCLUDE_DIR)

参考：

[Ubuntu / Windows下安装Libtiff库_tiff库下载-CSDN博客](https://blog.csdn.net/qq_30354455/article/details/90757239)

[opencv编译问题处理集_no package 'libdc1394-2' found-CSDN博客](https://blog.csdn.net/weixin_34910922/article/details/118095033)



["OpenCV is not able to find/configure CUDA SDK (required by WITH_CUDA)" when building CV4 - Jetson & Embedded Systems / Jetson Xavier NX - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opencv-is-not-able-to-find-configure-cuda-sdk-required-by-with-cuda-when-building-cv4/147870)



[ubuntu 安装ffmpeg，配置时出现 libvpx enabled but no supported decoders found，编译出现libavcoder.so有函数未定义问题的解决方法-CSDN博客](https://blog.csdn.net/weixin_42232238/article/details/106072886)
