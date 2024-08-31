# 复现AUW-GCN

项目github地址：  

[xjtupanda / AUW-GCN](https://github.com/xjtupanda/AUW-GCN)

## 复现过程

**项目环境**

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

参考pytorch版本官网：[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

### kaggle云服务器

使用kaggle上的GPU服务器环境为

> OS: Ubuntu 20.04.4 LTS  / Ubuntu 22.04.4 LTS
>
> Python: 3.10  
>
> CUDA: 12.1, cudnn: 8.5.0  / CUDA：12.3，cudnn：9.0.0
>
> GPU: NVIDIA Tesla P100 / NVIDIA Tesla T4×2

需要修改python版本和cuda版本

因为注册多个kaggle账号，早前注册的账号的服务器和CUDA的预置与目前的不同

### kaggle中的环境变量配置

使用`python`中的包`os`和`subprocess`可以保持每个命令行的环境变量一致

### 配置python环境

cuda默认的python版本为python3.10

对python版本进行修改

修改为**python3.8**，系统默认python3.8的版本为python3.8.13

使用conda进行环境的配置

```bash
# 配置python3.8环境
# Create New Conda Environment and Use Conda Channel 
!conda create -n newCondaEnvironment -c cctbx202208 python=3.8 -y
!source /opt/conda/bin/activate newCondaEnvironment && conda install -c cctbx202208 python=3.8 -y
!/opt/conda/envs/newCondaEnvironment/bin/python3 --version
!echo 'print("Hello, World!")' > test.py
!/opt/conda/envs/newCondaEnvironment/bin/python3 test.py
!sudo rm /opt/conda/bin/python3
!sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3
# !sudo rm /opt/conda/bin/python3.10
!sudo rm /opt/conda/bin/python3.10
!sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python3.8
# #rm: cannot remove '/opt/conda/lib/python3.10': Is a directory
# !sudo rm /opt/conda/lib/python3.10
# !sudo ln -sf /opt/conda/envs/newCondaEnvironment/lib/python3.8 /opt/conda/lib/python3.8
# #rm: cannot remove '/opt/conda/include/python3.10': Is a directory
# !sudo rm /opt/conda/include/python3.10
# !sudo ln -sf /opt/conda/envs/newCondaEnvironment/include/python3.8 /opt/conda/include/python3.8
!sudo rm /opt/conda/bin/python
!sudo ln -s /opt/conda/envs/newCondaEnvironment/bin/python3 /opt/conda/bin/python
# 使用pip进行依赖的安装
!sudo rm /opt/conda/bin/pip 
!sudo ln -sf /opt/conda/envs/newCondaEnvironment/bin/pip /opt/conda/bin/pip
```

在`requirement.txt`中`torch`的版本与`torchvision`的版本不一致

> Could not find a version that satisfies the requirement torchvision==0.11.2+cu102

1.将`torchvision`的版本改到与`torch`的版本一致

配置成`torchvision==0.14.1`

```bash
## pytorch 1.13.1
torch==1.13.1
torchvision==0.14.1
## 新增
torchaudio==0.13.1
```

这样配置是可行的，训练模型的代码也可以跑通

2.将`torch`的版本改到与`torchvision`的版本一致

配置成`torch==1.10.1`

```bash
!pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

使用这个版本会报错

```bash
File "/kaggle/working/AUW-GCN-test/train.py", line 7, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'
```

`distutils`模块在Python 3.10及更高版本中被逐步废弃并且部分功能不可再用

解决方法：降低python版本至3.8或者升高pytorch版本

由于已经配好python3.8

将pytorch版本置为pytorch==1.13.1

3.添加`torchaudio`

源项目中的`requirement.txt`文件中没有指定下载`torchaudio`

4.关于`nvidia-cublas-cu11`、`nvidia-cuda-nvrtc-cu11`、`nvidia-cuda-runtime-cu11`、`nvidia-cudnn-cu11`中的版本

```python
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
```

cu11是否代表cuda11x的环境？但是无论是kaggle的源环境cuda12x还是项目要求的cuda10.2环境都与cuda11不匹配

但是在进行模型训练时没有问题，不知道之后的数据集

### 配置CUDA环境

```bash
# 查看系统驱动版本
!nvidia-smi

Mon Aug 26 06:20:51 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P100-PCIE-16GB           Off |   00000000:00:04.0 Off |                    0 |
| N/A   40C    P0             26W /  250W |       0MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```bash
# 查看系统当前cuda的版本
!nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

#### 安装对应版本的gcc

直接安装cuda10.2，需要安装**对应版本的`gcc`**

会报错`Failed to verify gcc version. See log at /var/log/cuda-installer.log for details.`

参考[ubuntu22.10安装cuda出错Failed to verify gcc version. See log at /var/log/cuda-installer.log for details.-CSDN博客](https://blog.csdn.net/aizsa111/article/details/129455363)

[:: (nvidia.com)](https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/index.html)

这里不太清楚`g++`是否也需要匹配，但是还是匹配了`gcc`的版本

配置`gcc`和`g++`的版本为7.3.0，但是安装时只能指定7这样，不能指定小数

安装之后的版本为7.5.0，好像也能凑活

```bash
# 安装gcc-7 g++-7
!sudo apt-get install build-essential -y
!sudo apt-get -y install gcc-7 g++-7
```

安装完后需要激活，否则仍然是系统当前的gcc和g++的高版本

```bash
# 变换当前gcc g++版本 选择版本为7
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 20
!gcc --version
!g++ --version
```

检查系统中当前的cuda和cuDNN的版本

在kaggle中检查的相关命令参考：[Check CUDA and cuDNN (kaggle.com)](https://www.kaggle.com/code/titericz/check-cuda-and-cudnn)

使用kaggle时，一个账号的系统为ubuntu20.04，另外两个账号的系统为ubuntu22.04

使用的系统为ubuntu22.04时，没法下载gcc-7

比如回会报以下错误

```bash
E: Package 'gcc-7' has no installation candidate
E: Package 'g++-7' has no installation candidate
```

需要修改源文件`/etc/apt/sources.list`

```bash
# 添加源
deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe
```

Ubuntu清华镜像源网站：[ubuntu | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/)

参考：[[ubuntu\][原创]ubuntu22.04更换gcc版本为gcc-7_gcc11退回gcc7-CSDN博客](https://blog.csdn.net/FL1623863129/article/details/127796802)

将Ubuntu22.04在清华镜像源网站上的代码复制下载重新创建一个文件，再将focal源加入

即新创建的`sources.list`的内容如下

```bash
# ubuntu 22.04
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
# sudo dpkg  --print-architecture # 查看架构
# 输出的架构是 amd64
# [arch=amd64]
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
# # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse

# 用于下载 gcc-7 g++-7
deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe
```

由于在ubuntu22.04系统中运行，与ubuntu22.04系统进行区别，可对文件进行改名

```bash
# ubuntu 22.04
# 7~8min

#!rm AUW-GCN-test -rf
!git clone https://github.com/garlic0000/AUW-GCN-test.git
# 换成清华源
!sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
!sudo rm -rf /etc/apt/sources.list
# 火狐浏览器使用的kaggle用的是ubuntu22.04 没有gcc-7的下载
# 修改源文件 
!sudo cp /kaggle/working/AUW-GCN-test/other/sources_22_04.list /etc/apt/sources.list
!sudo apt-get update -y
!sudo apt-get upgrade -y
```

#### 安装CUDA

**安装cuda10.2**

各种版本的cuda下载链接：[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)

```bash
# 下载并安装cuda 10.2
!wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
!wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda_10.2.1_linux.run
!wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/2/cuda_10.2.2_linux.run
!sudo sh cuda_10.2.89_440.33.01_linux.run --silent --override-driver-check --toolkit --toolkitpath=/usr/local/cuda-10.2/ --librarypath=/usr/local/cuda-10.2/
!sudo sh cuda_10.2.1_linux.run --silent --override-driver-check
!sudo sh cuda_10.2.2_linux.run --silent --override-driver-check
# 下载日志
# !cp /var/log/cuda-installer.log /kaggle/working/cuda-installer.log
```

在安装的过程中不指定`toolkitpath`和`librarypath`会报错

参考:

[在kaggle中的notebook 如何自定义 cuda 版本以及如何使用自定义的conda或python版本运行项目（一）_kaggle cuda-CSDN博客](https://blog.csdn.net/Magicapprentice/article/details/139148080)

[cuda安装Installation failed log: [ERROR\]: Unable to determine libdir-CSDN博客](https://blog.csdn.net/weixin_44633882/article/details/108635093)

[CUDA Installation Guide for Linux (nvidia.com)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

安装完成后检查安装情况

```bash
# 查看系统中安装的cuda
!ls -l /usr/local | grep cuda
# 查看当前 cuda版本
!nvcc --version
# 安装好cuda10.2后 显示的仍然没改过来 但是系统已经使用cuda10.2
```

因为kaggle的GPU系统在连接session时没法重启所以在检查时可能一时没转变过来

但是在使用时cuda的版本已经置为cuda10.2

以上代码运行时可以看到`cuda`指向的路径变为`/usr/local/cuda-10.2`

在没有主动安装cuda前，kaggle默认的cuda版本为12.3，即cuda指向的路径可能为`/usr/local/cuda-12.3`

#### 安装cuDNN

**安装cuDNN7.6.5**

各种版本的cuDNN下载链接：[cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)

cuDNN与cuda安装不同，无法使用网页链接下载安装包或者可执行文件

使用这条命令下载`!wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/cudnn-10.2-linux-x64-v7.6.5.32.tgz`

下载下来的tgz文件无法解压，因为下载下来的文件是登录或者注册的网络请求的网页文件

需要注册和登录NVIDIA账号之后，从浏览器里下载获得离线文件，上传至kaggle后再解压

```bash
# 安装cuDNN 7.6.5
# 没法使用wget进行下载 因为需要注册 下载下来的文件没法安装或者解压
# 使用离线文件
!cp /kaggle/input/cudnn7.6.5/pytorch/default/1/libcudnn7_7.6.5.32-1cuda10.2_amd64.deb /kaggle/working/libcudnn7_7.6.5.32-1cuda10.2_amd64.deb
!cp /kaggle/input/cudnn7.6.5/pytorch/default/1/libcudnn7-dev_7.6.5.32-1cuda10.2_amd64.deb /kaggle/working/libcudnn7-dev_7.6.5.32-1cuda10.2_amd64.deb
!cp /kaggle/input/cudnn7.6.5/pytorch/default/1/libcudnn7-doc_7.6.5.32-1cuda10.2_amd64.deb /kaggle/working/libcudnn7-doc_7.6.5.32-1cuda10.2_amd64.deb
#注意安装顺序
!sudo dpkg -i libcudnn7_7.6.5.32-1cuda10.2_amd64.deb
!sudo dpkg -i libcudnn7-dev_7.6.5.32-1cuda10.2_amd64.deb
!sudo dpkg -i libcudnn7-doc_7.6.5.32-1cuda10.2_amd64.deb
```

当cuda版本与cuDNN不匹配时会报错：`nvcc fatal : Unsupported gpu architecture ‘compute_30‘`

```bash
# 验证cuDNN 7.6.5 的安装
!cp -r /usr/src/cudnn_samples_v7/ /kaggle/working/
!cd  /kaggle/working/cudnn_samples_v7/mnistCUDNN && make clean && make && ./mnistCUDNN
# gpu cuda版本不匹配可能会报错 使用cuda10.2不会报错
# nvcc fatal   : Unsupported gpu architecture 'compute_30'
```

参考：[nvcc fatal : Unsupported gpu architecture ‘compute_30‘-CSDN博客](https://blog.csdn.net/ttjbmkjsjyyjbbyg/article/details/114801686)

但是不需要像参考网页中那样修改文件，只需按照官网上下载匹配版本的cuda和cuDNN即可

参考：

[Ubuntu 18.04安装 CUDA 10.1 、cuDNN 7.6.5、PyTorch1.3 - BooTurbo - 博客园 (cnblogs.com)](https://www.cnblogs.com/booturbo/p/11834661.html)

[immanuelvalencia/Cuda-10.2-Installation-Guide-For-Ubutu-20.04: Contains the complete guide for installing CUDA 10.02 for Ubuntu 20.04 LTS (github.com)](https://github.com/immanuelvalencia/Cuda-10.2-Installation-Guide-For-Ubutu-20.04)

### 安装ffmpeg

最初借鉴这个网站上的方法进行这些软件的安装，但是总是报错

[最全、最新安装 Denseflow 教程，安装 CUDA11.8、12.4 支持的 OpenCV 4.X【MCPRL】_安装denseflow-CSDN博客](https://blog.csdn.net/baihupleonly/article/details/139360191)

安装这些软件的脚本源码来自github地址: https://github.com/innerlee/setup

使用这个github上的脚本安装后，在安装ffmpeg时总是报错，报错的原因是检测不到这几个软件要么是这几个软件版本太低

于是在下载了这个github中的`zznasm.sh`、`yasm.sh`、`libx264.sh`、`libx265.sh`、`ffmpeg.sh`这几个脚本后

对脚本的内容进行修改，主要修改安装路径、下载链接等

#### 安装nasm

安装nasm时根据脚本进行修改的可执行代码如下：

```bash
# Found no assembler
# Minimum version is nasm-2.13
# If you really want to compile without asm, configure with --disable-asm.
!wget https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz
!tar zxvf nasm-2.16.03.tar.gz nasm-2.16.03
# configure: WARNING: unrecognized options: --enable-shared
!cd nasm-2.16.03 && ./configure --prefix=/usr/local && make -j"$(nproc)" && make install 
```

但是执行这些代码后需要配置环境路径之类

#### 安装yasm

安装yasm时根据脚本进行修改的可执行代码如下：

```bash
# nasm/yasm not found or too old. Use --disable-x86asm for a crippled build.
!wget http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz
!tar zxvf yasm-1.3.0.tar.gz yasm-1.3.0
# configure: WARNING: unrecognized options: --enable-shared
!cd yasm-1.3.0 && ./configure --prefix=/usr/local && make -j"$(nproc)" && make install 
```

执行这些代码后仍然需要配置环境路径之类

#### 安装libx264

安装libx264时根据脚本进行修改的可执行代码如下：

```bash
# ERROR: libx264 not found
!git clone https://code.videolan.org/videolan/x264.git
# ERROR: x264 not found using pkg-config
!cd x264 && export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:"$PKG_CONFIG_PATH \
    && ./configure --prefix=/usr/local --enable-shared && make -j"$(nproc)" && make install
```

执行这些代码后仍然需要配置环境路径之类

#### 安装libx265

安装libx265时根据脚本进行修改的可执行代码如下：

```bash
!sudo git clone https://github.com/videolan/x265.git
!cd x265/build/linux && cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_SHARED:bool=on ../../source && make -j"$(nproc)" && make install
```

这个代码的版本与安装的ffmpeg的版本不匹配需要修改



执行这些代码后仍然需要配置环境路径之类

#### 安装libvpx

安装libvpx时根据脚本进行修改的可执行代码如下：

```bash
!sudo apt-get update -y
!sudo apt-get upgrade -y
# 已经安装过yasm 还是报错缺失yasm
!sudo apt-get install libtool pkg-config autoconf automake -y
!git clone https://github.com/webmproject/libvpx.git
!cd libvpx && ./configure --prefix=/usr/local \
    --disable-examples \
    --disable-unit-tests \
    --enable-vp9-highbitdepth \
    --as=yasm \
    --enable-shared && make -j"$(nproc)" && make install 
# make[1]: Nothing to be done for 'install'.
```

执行这些代码后仍然需要配置环境路径之类

运行完这些代码后，会提示

```bash
make[1]: Nothing to be done for 'install'.
```

不知道是什么原因，但是不影响ffmpeg的安装

#### 去掉libcurl.so.4警告信息

一定要去掉这个警告信息，不然在ffmpeg的编译安装和opencv的编译安装会不停的警告

```bash
# 去掉警告信息
# /usr/bin/cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)
# !ldd /usr/bin/cmake
# !cd /opt/conda/lib/ && ls
!rm -rf /opt/conda/lib/libcurl.so.4
```

#### 安装ffmpeg

安装ffmpeg时根据脚本进行修改的可执行代码如下：

```bash
!git clone https://git.ffmpeg.org/ffmpeg.git
!cd ffmpeg && export CFLAGS="-I/usr/local/include" \
    && export CPPFLAGS="-I/usr/local/include" \
    && export LDFLAGS="-L/usr/local/lib" \
    && export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:"$PKG_CONFIG_PATH

!cd ffmpeg && ./configure --prefix=/usr/local \
    --extra-libs=-lpthread \
    --extra-libs=-lm \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-nonfree \
    --enable-pic \
    --enable-shared && make -j$(nproc) && sudo make install
```

但是在运行时，仍然会提示找不到文件，明明已经下载，但是还是找不到文件

就算写入系统变量，还是提示找不到文件

原因是使用`!`运行命令和代码，自己设置的环境变量不会写入到系统中，甚至在运行下一个命令行时，系统变量又恢复如初

解决方法：

需要使用python库os和subprocess

并且把以上代码写在同一个命令块中，代码如下

```python
# 安装nasm, yasm, libx264, libx265, libvpx
# 安装ffmpeg
# 13~15min

import os
import subprocess

# 更新和升级系统
subprocess.run(['sudo', 'apt-get', 'update', '-y'], check=True)
subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'], check=True)

# 安装依赖
subprocess.run(['sudo', 'apt-get', 'install', 'libtool', 'pkg-config', 'autoconf', 'automake', '-y'], check=True)

# 下载并安装 nasm
nasm_url = 'https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz'
subprocess.run(['wget', nasm_url], check=True)
subprocess.run(['tar', 'zxvf', 'nasm-2.16.03.tar.gz'], check=True)

os.chdir('nasm-2.16.03')
subprocess.run(['./configure', '--prefix=/usr/local'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录

# 下载并安装 yasm
yasm_url = 'http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz'
subprocess.run(['wget', yasm_url], check=True)
subprocess.run(['tar', 'zxvf', 'yasm-1.3.0.tar.gz'], check=True)

os.chdir('yasm-1.3.0')
subprocess.run(['./configure', '--prefix=/usr/local'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录

# 克隆并安装 x264
subprocess.run(['git', 'clone', 'https://code.videolan.org/videolan/x264.git'], check=True)

os.chdir('x264')
os.environ['PKG_CONFIG_PATH'] = "/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

subprocess.run(['./configure', '--prefix=/usr/local', '--enable-shared'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录

# 克隆并安装 x265
# 使用github:https://github.com/videolan/x265.git上下载的文件版本与ffmpeg的版本不匹配
# 去https://bitbucket.org/multicoreware/x265_git/downloads/上下载最新版本
# 最新版好像也不行
#x265_url = 'https://bitbucket.org/multicoreware/x265_git/downloads/x265_3.6.tar.gz'
x265_url = 'https://bitbucket.org/multicoreware/x265_git/downloads/x265_3.5.tar.gz'
subprocess.run(['wget', x265_url], check=True)
subprocess.run(['tar', 'zxvf', 'x265_3.5.tar.gz'], check=True)

os.chdir('x265_3.5/build/linux')
os.environ['PKG_CONFIG_PATH'] = "/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

subprocess.run(['cmake', '-G', 'Unix Makefiles', '-DCMAKE_INSTALL_PREFIX=/usr/local', '-DENABLE_SHARED:bool=on', '../../source'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
subprocess.run(['ls'], check=True)
os.chdir('../../..')  # 返回上上上级目录
print('输出当前目录')
subprocess.run(['ls'], check=True)

# 确保 pkg-config 能找到 x265
os.environ['PKG_CONFIG_PATH'] = "/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

print('输出是否存在x265.pc文件')
subprocess.run(['ls', '/usr/local/lib/pkgconfig/x265.pc'])
# 验证 x265 是否安装成功
print("验证x265是否安装成功")
subprocess.run(['pkg-config', '--modversion', 'x265'], check=True) 

# 去掉警告信息
# /usr/bin/cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)
# !ldd /usr/bin/cmake
# !cd /opt/conda/lib/ && ls
#!rm -rf /opt/conda/lib/libcurl.so.4
# 设置需要删除的文件路径
file_to_remove = os.path.join(os.environ.get('CONDA_PREFIX', '/opt/conda'), 'lib', 'libcurl.so.4')
# 打印要删除的文件路径以确认
print(f"File to remove: {file_to_remove}")
# 使用 subprocess 来删除文件
try:
    subprocess.run(['rm', '-rf', file_to_remove], check=True)
    print("File removed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error removing file: {e}")

    
# 克隆并安装 libvpx
subprocess.run(['git', 'clone', 'https://github.com/webmproject/libvpx.git'], check=True)

os.chdir('libvpx')
os.environ['PKG_CONFIG_PATH'] = "/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

subprocess.run(['./configure', 
                '--prefix=/usr/local', 
                '--disable-examples', 
                '--disable-unit-tests', 
                '--enable-vp9-highbitdepth', 
                '--as=yasm', 
                '--enable-shared'
               ], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录

# ffmpeg: error while loading shared libraries: libavdevice.so.61: cannot open shared object file: No such file or directory
#!sudo ldconfig
subprocess.run(['sudo', 'ldconfig'], check=True)
# export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# 更新 LD_LIBRARY_PATH 环境变量
os.environ['LD_LIBRARY_PATH'] = "/usr/local/lib:" + os.environ.get('LD_LIBRARY_PATH', '')

# 安装ffnvcodec
subprocess.run(['git', 'clone', 'https://git.videolan.org/git/ffmpeg/nv-codec-headers.git'], check=True)
os.chdir('nv-codec-headers')
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录


# 克隆并安装 FFmpeg
subprocess.run(['git', 'clone', 'https://git.ffmpeg.org/ffmpeg.git'], check=True)

os.chdir('ffmpeg')
os.environ['CFLAGS'] = "-I/usr/local/include"
os.environ['CPPFLAGS'] = "-I/usr/local/include"
os.environ['LDFLAGS'] = "-L/usr/local/lib"
os.environ['PKG_CONFIG_PATH'] = "/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

subprocess.run([
    './configure',
    '--prefix=/usr/local',
    '--extra-libs=-lpthread',
    '--extra-libs=-lm',
    '--enable-gpl',
    '--enable-libx264',
    '--enable-libx265',
    '--enable-libvpx',
    '--enable-nonfree',
    '--enable-pic',
    '--enable-shared',
    '--extra-cflags=-I/usr/local/include',
    '--extra-ldflags=-L/usr/local/lib'
], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)

# 添加 /usr/local/lib 到 LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = "/usr/local/lib:" + os.environ.get('LD_LIBRARY_PATH', '')

# 验证 FFmpeg 安装
try:
    subprocess.run(['ffmpeg', '-codecs'], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error during FFmpeg verification: {e}")
subprocess.run(['sudo', 'ldconfig'], check=True)
subprocess.run(['ffmpeg', '-codecs'], check=True)
os.chdir('..')  # 返回上级目录
```

### 安装opencv

安装opencv4.10

系统环境设置为cuda10.2时

编译opencv时会提示cuda版本过低，应该使用cuda11以上的版本

1.cuda10.2 

2.cuda12.3

> -- Checking for module 'libavresample'
> --   No package 'libavresample' found

在`zzffmpeg.sh`中添加

`./configure --enable-libavresample`

添加的有问题

在新的ffmpeg中libavresample不再支持，被其他软件取代

### 安装denseflow

### 配置openGL

[TurboVNC+VirtualGL：实现服务器的多用户图形化访问与硬件加速 | 一颗栗子球 (shaoyecheng.com)](https://shaoyecheng.com/uncategorized/2020-04-08-TurboVNC-VirtualGL：实现服务器的多用户图形化访问与硬件加速.html)















### 上传项目代码、数据集、模型文件

在kaggle

### 数据集处理

数据集处理的环境与训练模型的环境不同。

数据集处理

### 训练和测试模型

配置好python环境和cuda环境后

安装项目所需要的python依赖

```python
# 下载项目代码并安装依赖
# 3~10min

# !rm AUW-GCN-test -rf
!git clone https://github.com/garlic0000/AUW-GCN-test.git
!cd AUW-GCN-test
# 无法使用pytorch 1.10.1 使用pytorch==1.13.1代替
# # 另外安装pytorch 1.10.1
# !pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
# 安装依赖
!pip install -r /kaggle/working/AUW-GCN-test/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

就可以进行模型的训练和测试

```bash
# 进行模型的训练和评估
# !rm -rf output
!bash /kaggle/working/AUW-GCN-test/pipeline.sh
```

### 结果下载

运行完成后进行结果下载

在文件`pipeline.sh`文件中配置的`OUTPUT`为`/kaggle/working/output`

由于运行结果为多个文件，在kaggle网页只能一个一个文件的下载，可进行以下脚本对输出文件进行打包，再下载压缩包

```python
# 将/kaggle/working/output文件夹压缩为output.zip
import os
import zipfile
import datetime

def file2zip(packagePath, zipPath):
    '''
  :param packagePath: 文件夹路径
  :param zipPath: 压缩包路径
  :return:
  '''
    zip = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirNames, fileNames in os.walk(packagePath):
        fpath = path.replace(packagePath, '')
        for name in fileNames:
            fullName = os.path.join(path, name)
            name = fpath + '\\' + name
            zip.write(fullName, name)
    zip.close()


if __name__ == "__main__":
    # 文件夹路径
    packagePath = '/kaggle/working/output'
    zipPath = '/kaggle/working/output.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    file2zip(packagePath, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
```

有时因为下载的文件过大，kaggle网页上点击下载按钮后无反应，可使用如下脚本进行下载

```python
# /kaggle/working/下的output.zip输出下载链接
import os
os.chdir('/kaggle/working')
print(os.getcwd())
print(os.listdir("/kaggle/working"))
from IPython.display import FileLink
FileLink('output.zip')
```



[【环境配置篇】保姆级教学之Ubuntu20.04上编译OpenCV+CUDA_ubuntu opencv cuda-CSDN博客](https://blog.csdn.net/ChunjieShan/article/details/125391238)









[【2022超详细版】Win10安装cuda（10.1、11.7）+cuDNN（7.6.5、8.5.0）+tensorflow(gpu版)+pytorch（gpu版）_cudnn7.6.5-CSDN博客](https://blog.csdn.net/m0_63834988/article/details/128781572)

[源码编译安装ffmpeg（带libx264安装）_yuanma bianyi anzhuang ffmepg-CSDN博客](https://blog.csdn.net/tl4832194/article/details/113857128)

[fatal error: gio/gio.h: 没有那个文件或目录 - CSDN文库](https://wenku.csdn.net/answer/0d2d12f8f9704a56bfe4cd616b29315b)

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

> 



["error: HAVE_INTROSPECTION does not appear in AM_CONDITIONAL" when compiling in Ubuntu · Issue #31 · solus-project/budgie-desktop (github.com)](https://github.com/solus-project/budgie-desktop/issues/31)
