

# 复现AUW-GCN实验报告

项目github地址：  

[xjtupanda / AUW-GCN](https://github.com/xjtupanda/AUW-GCN)

## 环境安装与配置

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

Ubuntu 20.04

> OS: Ubuntu 20.04.6 LTS
>
> Python: 3.10  
>
> CUDA: 12.1, cudnn: 8.5.0 
>
> GPU: NVIDIA Tesla P100 

Ubuntu 22.04

> OS: Ubuntu 22.04.4 LTS
>
> Python: 3.10  
>
> CUDA：12.3，cudnn：9.0.0
>
> GPU: NVIDIA Tesla P100 

需要修改python版本和cuda版本

因为注册多个kaggle账号，早前注册的账号的服务器和CUDA的预置与目前的不同

所以使用的Kaggle服务器有多个版本，有一个Ubuntu20.04和两个ubuntu22.04的Kaggle云服务器账号

### kaggle中的环境变量配置

使用`python`中的包`os`和`subprocess`可以保持每个命令行的环境变量一致

防止出现当前命令块相关环境变量配置完成，在下一个命令块失效的问题，可进行以下配置

```python
os.environ['LD_LIBRARY_PATH'] = "/usr/local/lib:/usr/lib/x86_64-linux-gnu:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PKG_CONFIG_PATH'] = "/usr/lib/x86_64-linux-gnu/pkgconfig:" + \
                                "/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')
os.environ['INCLUDE_PATH'] = '/usr/local/include:/usr/include' + os.environ.get('INCLUDE_PATH', '')

os.environ['CPLUS_INCLUDE_PATH']="/usr/include/:" + os.environ.get('CPLUS_INCLUDE_PATH', '')
```

使用Python语言运行shell命令，可以使用以下类似代码

```python
import subprocess

# 更新和升级系统
subprocess.run(['sudo', 'apt-get', 'update', '-y'], check=True)
subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'], check=True)
# 安装依赖
subprocess.run(['sudo', 'apt-get', 'install', 'libtool', 'pkg-config', 'autoconf', 'automake', '-y'], check=True)

```

### 配置python环境

CUDA默认的python版本为python3.10

因此需要对python版本进行修改，修改为**python3.8**。并且系统默认python3.8的版本为python3.8.13。

在Kaggle服务器上使用conda安装创建了虚拟环境并配置python3.8之后，系统的python版本仍然为python3.10。

解决办法是在cuda创建虚拟环境后，将虚拟环境的python用于真实环境的python。

即将真实环境的python可执行文件删除，再创建虚拟环境的python的可执行文件的软链接并连接到真实环境的python可执行文件。

以下是使用Kaggle自带的conda进行环境的配置：

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

这样操作下来会无法再使用conda，因此使用pip进行python软件的安装和管理。

**注意**

这样下来会破坏kaggle的内核。在进行save version连续运行时会有问题。运行到最后系统会检查内核，只有内核完整才能成功运行完成，并保存结果。

对上面的代码的更改

```bash
# 配置python3.8环境
# Create New Conda Environment and Use Conda Channel 
!conda create -n newCondaEnvironment -c cctbx202208 python=3.8 -y
!source /opt/conda/bin/activate newCondaEnvironment && conda install -c cctbx202208 python=3.8 -y
# 使用/opt/conda/envs/newCondaEnvironment/bin/python 运行python脚本
!/opt/conda/envs/newCondaEnvironment/bin/python --version

# 使用/opt/conda/envs/newCondaEnvironment/bin/pip进行依赖的安装
!/opt/conda/envs/newCondaEnvironment/bin/pip --version
!/opt/conda/envs/newCondaEnvironment/bin/pip install --upgrade pip 
```

关于`-c cctbx202208`

系统中提示使用`conda update -n base -c conda-forge conda`来升级conda，这里也有一个`-c`参数

这两有什么不一样？使用`-c conda-forge`能否安装python3.8？

`-c`是`channel`的缩写，在conda中，`channel`是包的存储位置，可以视为软件仓库。指定`-c`选项，就是告诉conda从特定的channel中搜索和安装包。对于安装特定版本的软件或从第三方源安装软件非常有用。

关于项目中requirements.txt的说明：

**1.torch版本问题**

在`requirement.txt`中`torch`的版本与`torchvision`的版本不一致会提示如下错误

> Could not find a version that satisfies the requirement torchvision==0.11.2+cu102

（1）将`torchvision`的版本改到与`torch`的版本一致

即配置成`torchvision==0.14.1`

将项目中requirements.txt中的依赖版本修改如下：

```bash
## pytorch 1.13.1
torch==1.13.1
torchvision==0.14.1
## 新增
torchaudio==0.13.1
```

这样配置是可行的，在实际运行中训练模型的代码可以跑通

在requirements.txt中这样写，和使用这样的命令安装的相同吗？

```bash
!pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

（2）将`torch`的版本改到与`torchvision`的版本一致

即配置成`torch==1.10.1`

```bash
!pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

但是使用这个版本会报错

```bash
File "/kaggle/working/ME-GCN-Project/train.py", line 7, in <module>
    from torch.utils.tensorboard import SummaryWriter
  File "/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
    LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'
```

这是因为`distutils`模块在Python 3.10及更高版本中被逐步废弃并且部分功能不可再用

解决方法：降低python版本至3.8或者升高pytorch版本

由于已经配好python3.8

将pytorch版本置为pytorch==1.13.1

**2.添加`torchaudio`**

源项目中的`requirement.txt`文件中没有指定下载`torchaudio`

可以在requirement.txt文件中添加，并指定版本`torchaudio==0.13.1`

**3.关于`nvidia-cublas-cu11`、`nvidia-cuda-nvrtc-cu11`、`nvidia-cuda-runtime-cu11`、`nvidia-cudnn-cu11`中的版本**

```python
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
```

cu11是否代表cuda11x的环境？但是无论是kaggle的源环境cuda12x还是项目要求的cuda10.2环境都与cuda11不匹配

但是在进行模型训练时没有问题，不知道关于的数据集的预处理和特征提取部分是否会有影响。

在下载cuDNN的页面，有cuDNN8.5.0用于cuda11.7和cuda10.2

**4.在Ubuntu 22.04 环境中，使用python3.8，更改requirements.txt，不指定安装软件的版本**

关于

```
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
```

修改为

```
nvidia-cublas-cu12
nvidia-cuda-nvrtc-cu12
nvidia-cuda-runtime-cu12
nvidia-cudnn-cu12
```

报错

> pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='pypi.tuna.tsinghua.edu.cn', port=443): Read timed out.

进行如下修改，可解决

```sh
!pip install --upgrade pip
!pip install -r /kaggle/working/ME-GCN-Project/requirements_no_version.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100


```

**5.使用kaggle原生环境cuda12.x和python3.10不知是否可行**

经过测试是可行的

ubuntu22.04 python3.10 cuda12.3 完全可行

requirements.txt的不写版本号

再进行这样的更改

```
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
```

修改为

```
nvidia-cublas-cu12
nvidia-cuda-nvrtc-cu12
nvidia-cuda-runtime-cu12
nvidia-cudnn-cu12
```

其中有警告，不影响正常运行，警告内容为

```bash
/opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
```

因为`os.fork()`在使用多线程代码时可能会导致死锁，特别是与JAX（加速线性代数库）的多线程工作原理冲突。

使用`spawn`而不是`fork`，python的`multiprocessing` 库默认使用`fork`来创建子进程，但在多线程应用中，可能会出现问题，通过设置`multiprocessing` 的启动方式为`spawn`来避免这种问题

在代码的入口出添加以下代码：

```python
import multiprocessing

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
```

这样修改后非常的慢

在 Python 的 `multiprocessing` 模块中，不同的启动方式（start methods）用于创建新进程。主要的启动方式有三种：`fork`、`spawn` 和 `forkserver`。它们的工作方式和使用场景各不相同。下面解释 `spawn` 和 `fork` 以及它们的区别：

1. **fork**

`fork` 是 Unix（包括 Linux 和 macOS）系统上的默认启动方式。

- **工作原理**：`fork` 通过复制父进程的内存空间来创建子进程。子进程几乎是父进程的完整副本，继承了父进程的全局状态（包括所有内存中的数据、代码、打开的文件描述符等）。
- 优点：
  - 启动速度快，因为不需要重新加载 Python 解释器或模块。
  - 子进程继承父进程的状态，可以直接访问父进程的全局变量。
- 缺点：
  - 如果父进程使用了多线程，`fork` 可能会导致一些问题，比如死锁，特别是在使用多线程或像 `JAX` 这样的库时。
  - 某些资源的继承可能会导致不可预知的行为，特别是在跨平台代码中。

2. **spawn**

`spawn` 是 Windows 系统上的默认启动方式，也可用于 Unix 系统。

- **工作原理**：`spawn` 创建一个全新的 Python 解释器实例，并通过导入模块和运行特定代码来启动进程。父进程不会自动将任何全局状态传递给子进程，所有需要的数据都必须通过显式的方式（如通过函数参数）传递给子进程。
- 优点：
  - 适合多线程程序，因为它不会继承父进程的线程状态，避免了 `fork` 导致的死锁或不稳定性问题。
  - 由于不会继承父进程的全局状态，更适合跨平台代码，尤其是在 Windows 上。
- 缺点：
  - 启动速度较慢，因为每个新进程都需要重新加载 Python 解释器和所有依赖模块。
  - 需要显式传递父进程的数据，无法直接访问父进程的全局变量。

3. **forkserver**

`forkserver` 是在 Unix 上的一种可选方式，不是默认的。

- **工作原理**：当你使用 `forkserver` 时，Python 会启动一个守护进程（fork server），然后通过该守护进程来生成新的子进程。守护进程通过 `fork` 来创建进程，因此继承了最初父进程的状态，但新的进程不会直接从当前父进程中 `fork`。
- 优点：
  - 避免了多线程和 `fork` 可能带来的问题。
  - 比 `spawn` 启动更快，因为进程是从守护进程而不是完全新建的环境中生成的。
- 缺点：
  - 守护进程始终保持运行，可能会占用资源。
  - 仅在支持 `fork` 的 Unix 系统上可用，Windows 上不可用。

**总结：`fork` vs `spawn`**

- **`fork`**：更快，继承父进程的状态，但在多线程环境中可能不安全。
- **`spawn`**：更安全，适合多线程和跨平台代码，但启动较慢，因为它会启动全新的进程。

在使用 JAX 等多线程库时，使用 `spawn` 启动方式通常更安全。

### 配置CUDA环境

查看系统驱动版本

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

查看系统当前cuda版本

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

安装之后的版本为7.5.0，也能支持下载cuda10.2

```bash
# 安装gcc-7 g++-7
!sudo apt-get install build-essential -y
!sudo apt-get -y install gcc-7 g++-7
```

安装完后需要激活，否则仍然是系统当前的gcc和g++的高版本

变换版本的命令如下：

```bash
# 变换当前gcc g++版本 选择版本为7
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 20
!gcc --version
!g++ --version
```

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

#!rm ME-GCN-Project -rf
!git clone https://github.com/garlic0000/ME-GCN-Project.git
# 换成清华源
!sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
!sudo rm -rf /etc/apt/sources.list
# 火狐浏览器使用的kaggle用的是ubuntu22.04 没有gcc-7的下载
# 修改源文件 
!sudo cp /kaggle/working/ME-GCN-Project/other/sources_22_04.list /etc/apt/sources.list
!sudo apt-get update -y
!sudo apt-get upgrade -y
```

#### 安装CUDA

**安装cuda10.2**

检查系统中当前的cuda和cuDNN的版本

在kaggle中检查的相关命令参考：[Check CUDA and cuDNN (kaggle.com)](https://www.kaggle.com/code/titericz/check-cuda-and-cudnn)

各种版本的cuda下载链接：[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)

下载并安装cuda 10.2的命令如下

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

toolkitpath是软件安装的地址，默认为`/usr/local/cuda-10.2`

参考:

[在kaggle中的notebook 如何自定义 cuda 版本以及如何使用自定义的conda或python版本运行项目（一）_kaggle cuda-CSDN博客](https://blog.csdn.net/Magicapprentice/article/details/139148080)

[cuda安装Installation failed log: [ERROR\]: Unable to determine libdir-CSDN博客](https://blog.csdn.net/weixin_44633882/article/details/108635093)

[CUDA Installation Guide for Linux (nvidia.com)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

安装完成后检查安装情况，命令如下

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

即`cuda -> /usr/local/cuda-10.2`

在没有主动安装cuda前，kaggle默认的cuda版本为12.3，即cuda指向的路径可能为`/usr/local/cuda-12.3`

**关于`!nvcc --version`**

安装完新版本的cuda之后，执行这条命令后输出的结果不是新安装的cuda的版本

比如系统默认cuda12.3 新安装cuda11.7

```bash
!nvcc --version
!/usr/local/cuda/bin/nvcc --version
-----------------------------------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```

查找nvcc的位置

```
!which nvcc
-----------------
/opt/conda/bin/nvcc
```

但是

```
!/opt/conda/bin/nvcc --version
-------------------------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

对其进行修改

```
!rm -rf /opt/conda/bin/nvcc
!ln -sf /usr/local/cuda/bin/nvcc /opt/conda/bin/nvcc
```

再输出

```
!/opt/conda/bin/nvcc --version
-------------------------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_18:49:52_PDT_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```



**安装cuda12.1**

cuda10.2在ubuntu22.04中使用起来很不方便

```bash
# 下载并安装cuda12.1
!wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
!sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.1/ 
```

安装cuda12.1也无法解决cmake无法识别系统架构的问题

#### 安装cuDNN

**安装cuDNN7.6.5**

各种版本的cuDNN下载链接：[cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)

cuDNN与cuda安装不同，无法使用网页链接下载安装包或者可执行文件

使用这条命令无法下载

```bash
!wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/cudnn-10.2-linux-x64-v7.6.5.32.tgz
```

下载下来的tgz文件无法解压，因为下载下来的文件是登录或者注册的网络请求的网页文件

需要注册和登录NVIDIA账号之后，从浏览器里下载获得离线文件，上传至kaggle后再解压

这里的操作是将deb软件包上传至模型目录后复制到当前工作目录，即`/kaggle/working/`下

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

验证cuDNN 7.6.5 的安装，这里的验证可以同时检验cuda和cuDNN是否安装成功.

因为当前系统默认的cuda为12.3，这个版本与cuDNN7.6.5不匹配，在运行检验cuDNN的代码时会报以下错误：

`nvcc fatal : Unsupported gpu architecture ‘compute_30‘`

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

[【2022超详细版】Win10安装cuda（10.1、11.7）+cuDNN（7.6.5、8.5.0）+tensorflow(gpu版)+pytorch（gpu版）_cudnn7.6.5-CSDN博客](https://blog.csdn.net/m0_63834988/article/details/128781572)

**安装cuDNN8.5.0**

1.下载安装软件，不能直接通过链接进行下载，需要从浏览器登录后下载文件，然后将文件上传至kaggle服务器

```bash
# 无法使用链接下载
!cp /kaggle/input/cudnn8.5.0/pytorch/default/1/cudnn-local-repo-ubuntu2204-8.5.0.96_1.0-1_amd64.deb /kaggle/working/cudnn-local-repo-ubuntu2204-8.5.0.96_1.0-1_amd64.deb
!sudo dpkg -i cudnn-local-repo-ubuntu2204-8.5.0.96_1.0-1_amd64.deb
```

2.复制公钥

安装时会提示以下信息

> ```bash
> The public CUDA GPG key does not appear to be installed.
> To install the key, run this command:
> sudo cp /var/cudnn-local-repo-ubuntu2204-8.5.0.96/cudnn-local-7ED72349-keyring.gpg /usr/share/keyrings/
> ```

3.安装`libcudnn8`、`libcudnn8-dev`、`libcudnn8-samples`遇到了问题

```bash
!sudo apt-get install libcudnn8=8.5.0.96-1+cuda11.7 -y
!sudo apt-get install libcudnn8-dev=8.5.0.96-1+cuda11.7 -y
!sudo apt-get install libcudnn8-samples=8.5.0.96-1+cuda11.7 -y
```

使用上面这三个安装命令会报以下错误

> ```bash
> Some packages could not be installed. This may mean that you have
> requested an impossible situation or if you are using the unstable
> distribution that some required packages have not yet been created
> or been moved out of Incoming.
> The following information may help to resolve the situation:
> 
> The following packages have unmet dependencies:
>  libcudnn8-dev : Depends: libcudnn8 (= 8.5.0.96-1+cuda11.7) but 8.9.7.29-1+cuda12.2 is to be installed
> E: Unable to correct problems, you have held broken packages.
> Reading package lists... Done
> Building dependency tree... Done
> Reading state information... Done
> Package libcudnn8-samples is not available, but is referred to by another package.
> This may mean that the package is missing, has been obsoleted, or
> is only available from another source
> 
> E: Version '8.5.0.96-1+cuda11.7' for 'libcudnn8-samples' was not found
> Reading package lists... Done
> Building dependency tree... Done
> Reading state information... Done
> 0 upgraded, 0 newly installed, 0 to remove and 4 not upgraded.
> cp: cannot stat '/usr/src/cudnn_samples_v8/': No such file or directory
> /bin/bash: line 1: cd: /kaggle/working/cudnn_samples_v8/mnistCUDNN: No such file or directory
> ```

需要在以下路径进行安装

```bash
!cd /var/cudnn-local-repo-ubuntu2204-8.5.0.96/ && ls
-----------------------------------------------------------------------------
7ED72349.pub   Packages     cudnn-local-7ED72349-keyring.gpg
InRelease      Packages.gz  libcudnn8-dev_8.5.0.96-1+cuda11.7_amd64.deb
Local.md5      Release	    libcudnn8-samples_8.5.0.96-1+cuda11.7_amd64.deb
Local.md5.gpg  Release.gpg  libcudnn8_8.5.0.96-1+cuda11.7_amd64.deb
```

安装命令为

```bash
!cd /var/cudnn-local-repo-ubuntu2204-8.5.0.96/ && ls
!cd /var/cudnn-local-repo-ubuntu2204-8.5.0.96/ && sudo dpkg -i libcudnn8_8.5.0.96-1+cuda11.7_amd64.deb
!cd /var/cudnn-local-repo-ubuntu2204-8.5.0.96/ && sudo dpkg -i libcudnn8-dev_8.5.0.96-1+cuda11.7_amd64.deb
!cd /var/cudnn-local-repo-ubuntu2204-8.5.0.96/ && sudo dpkg -i libcudnn8-samples_8.5.0.96-1+cuda11.7_amd64.deb
```

参考网站：[【环境搭建】（五）Ubuntu22.04安装cuda_11.8.0+cudnn_8.6.0_cuda 11.8-CSDN博客](https://blog.csdn.net/weixin_41809117/article/details/137152503)

注意，安装完成之后不要使用

```bash
!sudo apt-get update -y
!sudo apt-get upgrade -y
```

因为，又重新升级了

```bash
cuDNN:                         YES (ver 8.9.7)
```

```
Preparing to unpack .../libcudnn8-dev_8.9.7.29-1+cuda12.2_amd64.deb ...
update-alternatives: removing manually selected alternative - switching libcudnn to auto mode
update-alternatives: using /usr/include/x86_64-linux-gnu/cudnn_v9.h to provide /usr/include/cudnn.h (libcudnn) in auto mode
Unpacking libcudnn8-dev (8.9.7.29-1+cuda12.2) over (8.5.0.96-1+cuda11.7) ...
Preparing to unpack .../libcudnn8_8.9.7.29-1+cuda12.2_amd64.deb ...
Unpacking libcudnn8 (8.9.7.29-1+cuda12.2) over (8.5.0.96-1+cuda11.7) ...
Preparing to unpack .../libcudnn8-samples_8.9.7.29-1+cuda12.2_amd64.deb ...
Unpacking libcudnn8-samples (8.9.7.29-1+cuda12.2) over (8.5.0.96-1+cuda11.7) ...
Setting up libcudnn8 (8.9.7.29-1+cuda12.2) ...
Setting up libcudnn8-dev (8.9.7.29-1+cuda12.2) ...
```

不知道有没有什么办法可以阻止

```
# 列出所有可用的cuDNN版本
!sudo update-alternatives --list libcudnn
# 指定需要的cuDNN版本
!sudo update-alternatives --set libcudnn /usr/lib/cudnn-8.5.0/libcudnn.so.8.5.0
# 验证当前的cuDNN版本
!cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

4.验证cuDNN 8.5.0的安装

```bash
# 验证cuDNN 8.5.0 的安装
# 验证cuDNN 8.5.0 的安装过程需要gcc9 g++9 否则会报错
!cp -r /usr/src/cudnn_samples_v8/ /kaggle/working/
!cd  /kaggle/working/cudnn_samples_v8/mnistCUDNN && make clean && make && ./mnistCUDNN
```

执行以上命令会有以下错误

> ```bash
> test.c:1:10: fatal error: FreeImage.h: No such file or directory
>     1 | #include "FreeImage.h"
>       |          ^~~~~~~~~~~~~
> compilation terminated.
> ```

安装FreeImage库

```bash
!sudo apt-get install libfreeimage-dev -y
```

需要安装gcc9和g++9 但是不确定之后编译opencv时 gcc和g++是否要换回

```bash
# 安装gcc-9 g++9
!sudo apt-get install build-essential -y
!sudo apt-get install gcc-9 -y
!sudo apt-get install g++-9 -y

# 变换当前gcc g++版本 选择版本为9
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20
!update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20
!gcc --version
!g++ --version
```

参考网站：[Ubuntu22.04安装Cuda11.3和Cudnn8.5的深度学习GPU环境_cudnn-download-survey-CSDN博客](https://blog.csdn.net/u014297502/article/details/126863549)

因为使用gcc-11和g++-11也可以安装成功

> ```bash
> cudnnGetVersion() : 8500 , CUDNN_VERSION from cudnn.h : 8500 (8.5.0)
> Host compiler version : GCC 11.4.0
> ......
> Test passed!
> ```

也就是这两个gcc和g++版本都可以完成

但是因为cuda11最高支持gcc的版本是gcc-10，所以使用gcc-9和gcc-10比较稳妥

使用gcc-11可能会出现兼容性问题

比如，报错

> ```bash
> In file included from /usr/local/cuda/include/crt/math_functions.h:10545,
>                  from /usr/local/cuda/include/crt/common_functions.h:303,
>                  from /usr/local/cuda/include/cuda_runtime.h:115,
>                  from <command-line>:
> /usr/include/c++/11/cmath:45:15: fatal error: math.h: No such file or directory
>    45 | #include_next <math.h>
>       |               ^~~~~~~~
> compilation terminated.
> make: *** [Makefile:241: fp16_dev.o] Error 1
> ```

使用这条命令也无法解决问题

```bash
sudo apt-get install libc6-dev -y
```

之所以安装cuDNN8.5.0是因为ubuntu22.04 cuda12.3 cuDNN9.0无法检测当前GPU的架构

而之前使用的Ubuntu20.04 cuda12.1 cuDNN 8.5.0 可以自动检测GPU的架构

在安装完cuDNN且进行验证时，输出

> ```bash
> /usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_87,code=sm_87 -gencode arch=compute_87,code=compute_87 -o fp16_dev.o -c fp16_dev.cu
> nvcc warning : The 'compute_35', 'compute_37', 'sm_35', and 'sm_37' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
> ```

目前系统的GPU为 Tesla P100  满足的架构为`compute_60`

### 安装ffmpeg

最初借鉴这个网站上的方法进行这些软件的安装，但是总是报错

[最全、最新安装 Denseflow 教程，安装 CUDA11.8、12.4 支持的 OpenCV 4.X【MCPRL】_安装denseflow-CSDN博客](https://blog.csdn.net/baihupleonly/article/details/139360191)

安装这些软件的脚本源码来自github地址: https://github.com/innerlee/setup

使用这个github上的脚本安装后，在安装ffmpeg时总是报错，报错的原因是检测不到这几个软件要么是这几个软件版本太低

于是在下载了这个github中的`zznasm.sh`、`zzyasm.sh`、`zzlibx264.sh`、`zzlibx265.sh`、`zzffmpeg.sh`这几个脚本后

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

需要使用以下链接进行下载

```bash
# 克隆并安装 x265
# 使用github:https://github.com/videolan/x265.git上下载的文件版本与ffmpeg的版本不匹配
# 去https://bitbucket.org/multicoreware/x265_git/downloads/上下载最新版本
# 最新版好像也不行
#x265_url = 'https://bitbucket.org/multicoreware/x265_git/downloads/x265_3.6.tar.gz'
x265_url = 'https://bitbucket.org/multicoreware/x265_git/downloads/x265_3.5.tar.gz'
```

从github上下载编译安装的版本与ffmpeg不匹配，ffmpeg在编译安装时始终无法检测到x265

需要从bitbucket网站上下载，而且不能下载最新版，最新是2024年发布，与安装的ffmpeg也不匹配，安装3.5版的可行

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

> make[1]: Nothing to be done for 'install'.

不知道是什么原因，但是不影响ffmpeg的安装

#### 去掉libcurl.so.4警告信息

一定要去掉这个警告信息，不然在ffmpeg的编译安装和opencv的编译安装会不停的警告

> ```bash
> x265_3.5/x265Version.txt
> -- cmake version 3.22.1
> cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by cmake)
> ```

> /usr/bin/cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)

在`/opt/conda/lib/`路径下有三个这样的文件，删除`libcurl.so.4`

> ```
> libcurl.so
> libcurl.so.4
> libcurl.so.4.8.0
> ```

```bash
# 去掉警告信息
# /usr/bin/cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)
# !ldd /usr/bin/cmake
# !cd /opt/conda/lib/ && ls
!rm -rf /opt/conda/lib/libcurl.so.4
```

参考网站：[linux cmake error no version information available - HappyCoder_1 - 博客园 (cnblogs.com)](https://www.cnblogs.com/132818Creator/p/13091631.html)

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

原因是在Kaggle命令块中使用`!`运行命令和代码，自己设置的环境变量不会写入到系统中，甚至在运行下一个命令行时，系统变量又恢复如初

解决方法：

需要使用python库os和subprocess

使用`os.environ`进行各种环境变量的设置

以及使用`subprocess.run`运行各种bash命令，保持不同命令块的环境相同

对上面的命令分别进行修改为：

1.下载并编译nasm

```bash
# 下载并安装 nasm

import os
import subprocess

nasm_url = 'https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.gz'
subprocess.run(['wget', nasm_url], check=True)
subprocess.run(['tar', 'zxvf', 'nasm-2.16.03.tar.gz'], check=True)

os.chdir('nasm-2.16.03')
subprocess.run(['./configure', '--prefix=/usr/local'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录
```

2.下载并编译yasm

```bash
# 下载并安装 yasm
import os
import subprocess

yasm_url = 'http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz'
subprocess.run(['wget', yasm_url], check=True)
subprocess.run(['tar', 'zxvf', 'yasm-1.3.0.tar.gz'], check=True)

os.chdir('yasm-1.3.0')
subprocess.run(['./configure', '--prefix=/usr/local'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录
```

3.下载并编译x264

```bash
# 克隆并安装 x264
subprocess.run(['git', 'clone', 'https://code.videolan.org/videolan/x264.git'], check=True)

os.chdir('x264')
os.environ['PKG_CONFIG_PATH'] = "/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

subprocess.run(['./configure', '--prefix=/usr/local', '--enable-shared'], check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], check=True)
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录
```

4.下载并编译x265

```python
# 克隆并安装 x265
import os
import subprocess
# 更新和升级系统
subprocess.run(['sudo', 'apt-get', 'update', '-y'], check=True)
subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'], check=True)

# 安装依赖
subprocess.run(['sudo', 'apt-get', 'install', 'libtool', 'pkg-config', 'autoconf', 'automake', '-y'], check=True)
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
```

5.下载并编译libvpx

```python
# 克隆并安装 libvpx
import os
import subprocess
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
```

6.去掉libcurl.so警告信息

```python
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
```

7.安装ffmpeg

```python
# 克隆并安装 FFmpeg
import os
import subprocess
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
# ffmpeg: error while loading shared libraries: libavdevice.so.61: cannot open shared object file: No such file or directory
#!sudo ldconfig
subprocess.run(['sudo', 'ldconfig'], check=True)
# export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# 更新 LD_LIBRARY_PATH 环境变量
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

按照以上顺序即可安装ffmpeg，ffmpeg是安装opencv的基础和前提。

使用apt-get的方式安装

```bash
!sudo apt-get update -y
!sudo apt-get upgrade -y
# 安装nasm yasm
!sudo apt-get install nasm yasm -y
# 安装x264 x265
!sudo apt-get install libx264-dev libx265-dev -y
# 安装 libvpx
!sudo apt-get install libvpx-dev -y
# 安装ffmpeg
!sudo apt-get install ffmpeg -y
```

注意要更新同步环境变量

检测安装

```bash
!nasm -v
!yasm --version
# No package 'x264' found
!pkg-config --modversion x264
!pkg-config --modversion x265
# No package 'vpx' found
!pkg-config --modversion vpx
!ffmpeg -codecs | grep libx264
!ffmpeg -codecs | grep libx265
!ffmpeg -codecs | grep libvpx
!ffmpeg -version
```



### 配置openGL

> ```bash
> OpenGL support:              
> YES (/usr/lib/x86_64-linux-gnu/libOpenGL.so /usr/lib/x86_64-linux-gnu/libGLX.so /usr/lib/x86_64-linux-gnu/libGLU.so)
> ```

不知道啥情况就支持了

编译安装

```bash
!git clone https://gitlab.gnome.org/Archive/gtkglext.git
!cd gtkglext && ./autogen.sh && make && sudo make install
```

还创建了软链接

```bash
# !ln -sf /usr/bin/glib-mkenums /usr/lib/bin/glib-mkenums
import subprocess
# 删除现有的符号链接或目录（如果存在）
subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/bin'], check=True)
# 创建新的符号链接
subprocess.run(['sudo', 'ln', '-s', '/usr/bin', '/usr/lib/bin'], check=True)

# 检查符号链接的目标路径
result = subprocess.run(['readlink', '-f', '/usr/lib/bin'], capture_output=True, text=True)
print(f"/usr/lib/bin points to: {result.stdout.strip()}")
```

但是报错了

> ```bash
> make[4]: *** No rule to make target 'GdkGLExt-1.0.typelib', needed by 'all-am'.  Stop.
> make[4]: Leaving directory '/kaggle/working/gtkglext/gdk'
> make[3]: *** [Makefile:933: all-recursive] Error 1
> make[3]: Leaving directory '/kaggle/working/gtkglext/gdk'
> make[2]: *** [Makefile:670: all] Error 2
> make[2]: Leaving directory '/kaggle/working/gtkglext/gdk'
> make[1]: *** [Makefile:574: all-recursive] Error 1
> make[1]: Leaving directory '/kaggle/working/gtkglext'
> make: *** [Makefile:475: all] Error 2
> ```

与这个报错信息一起的还有openGL相关的信息

不知道是否与从源码编译安装gtkglext有关？

还需要测试，各种依赖包已经安装完。

测试完后是无关的。



在安装openGL的过程中发现Kaggle服务器没有显示器，不支持安装openGL

但是可以安装VirtualGL代替运行openGL的程序

由于目前还没测试，不知这样的操作是否可行

#### 安装VirtualGL

安装VirtualGL的代码如下

```bash
# TurboVNC+VirtualGL
# https://shaoyecheng.com/uncategorized/2020-04-08-TurboVNC-VirtualGL%EF%BC%9A%E5%AE%9E%E7%8E%B0%E6%9C%8D%E5%8A%A1%E5%99%A8%E7%9A%84%E5%A4%9A%E7%94%A8%E6%88%B7%E5%9B%BE%E5%BD%A2%E5%8C%96%E8%AE%BF%E9%97%AE%E4%B8%8E%E7%A1%AC%E4%BB%B6%E5%8A%A0%E9%80%9F.html
# E: Package 'virtualgl' has no installation candidate
!wget https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb
!sudo dpkg -i virtualgl_3.1_amd64.deb
!sudo apt-get install -f -y
!vglrun +v
```

配置virtualGL时出错

由于Kaggle服务器中的命令行无法进行交互

但是以下代码必须进行交互

```bash
!which vglrun
!which vglserver_config
!/usr/bin/vglserver_config
# 使用脚本执行 避免交互
```

#### 安装TurboVNC

在安装TurboVNC报错

主要是运行以下代码报错

```bash
!nvidia-xconfig --query-gpu-info
!sudo nvidia-xconfig -a --allow-empty-initial-configuration --use-display-device=None --virtual=1920x1080 --busid {PCI:0:4:0}
```

无法创建虚拟显示器

至此无论是openGL还是TurboVNC和VirtualGL都无法安装和配置成功

如果之后的代码一定需要使用openGL，则需要租用服务器

参考网站：

[TurboVNC+VirtualGL：实现服务器的多用户图形化访问与硬件加速 | 一颗栗子球 (shaoyecheng.com)](https://shaoyecheng.com/uncategorized/2020-04-08-TurboVNC-VirtualGL：实现服务器的多用户图形化访问与硬件加速.html)

### 安装opencv

安装opencv时遇到了非常多的问题，到目前为止，问题还没解决。

以下是已经安装成功的步骤。

#### 安装Julia

使用软件包安装Julia，无法找到include包，因此需要进行源码编译安装。

```python
# 安装 Julia
import os
import subprocess

subprocess.run(['wget', 'https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz'], check=True)
subprocess.run(['tar', '-xvzf', 'julia-1.7.3-linux-x86_64.tar.gz'], check=True)
subprocess.run(['sudo', 'mv', 'julia-1.7.3', '/usr/local/'], check=True)
os.environ['PATH'] = "/usr/local/julia-1.7.3/bin:" + os.environ.get('PATH', '')
julia_install_command = """
using Pkg
Pkg.add("CxxWrap")
"""
subprocess.run(['julia', '-e', julia_install_command], check=True)
```

#### 安装ffnvcodec

安装ffnvcodec主要是为安装Video_Codec_SDK_10.0.26做准备

```python
# 安装ffnvcodec
import os
import subprocess

subprocess.run(['git', 'clone', 'https://git.videolan.org/git/ffmpeg/nv-codec-headers.git'], check=True)
os.chdir('nv-codec-headers')
subprocess.run(['sudo', 'make', 'install'], check=True)
os.chdir('..')  # 返回上级目录
```

#### 安装vulkan

在网站https://vulkan.lunarg.com/sdk/home上下载软件时注意查看当前服务器的系统版本

安装vulkan的代码如下

```bash
# ubuntu22.04 安装vulkan
# https://blog.csdn.net/weixin_43442574/article/details/119541899
# import os
# import subprocess

# subprocess.run(['sudo', 'wget', '-qO-', 'https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc'], check=True)
# subprocess.run(['sudo', 'wget', '-qO-', '/etc/apt/sources.list.d/lunarg-vulkan-1.3.290-jammy.list', 'https://packages.lunarg.com/vulkan/1.3.290/lunarg-vulkan-1.3.290-jammy.list'], check=True)
# subprocess.run(['sudo', 'apt-get', 'update', '-y'], check=True)
# subprocess.run(['sudo', 'apt-get', 'install', 'vulkan-sdk', '-y'], check=True)

!wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
!sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.290-jammy.list https://packages.lunarg.com/vulkan/1.3.290/lunarg-vulkan-1.3.290-jammy.list
!sudo apt-get update -y
!sudo apt-get install vulkan-sdk -y
```

#### 安装Video_Codec_SDK

Video Codec SDK多版本下载页面：[Video Codec SDK Archive | NVIDIA Developer](https://developer.nvidia.com/video-codec-sdk-archive)

> -- NVCUVENC: Header not found, WITH_NVCUVENC requires Nvidia encoding library header /usr/local/cuda;/usr/local/cuda/include/nvEncodeAPI.h

在[Video Codec SDK - Get Started | NVIDIA Developer](https://developer.nvidia.com/nvidia-video-codec-sdk/download)下载**Video Codec SDK 12.2**完整文件

将文件上传至服务器后，再将头文件复制到`/usr/local/cuda/include/`

```bash
!cp /kaggle/input/video_codec_sdk/pytorch/default/1/Video_Codec_SDK_12.2.72/Interface/* /usr/local/cuda/include/
```

> \- NVCUVENC: Library not found, WITH_NVCUVENC requires the Nvidia encoding shared library libnvidia-encode.so from the driver installation or the location of the stub library to be manually set with CUDA_nvidia-encode_LIBRARY i.e. CUDA_nvidia-encode_LIBRARY=/home/user/Video_Codec_SDK_X.X.X/Lib/linux/stubs/x86_64/libnvidia-encode.so

将库文件复制到`/usr/local/cuda/lib64/`

```sh
!cp /kaggle/input/video_codec_sdk/pytorch/default/1/Video_Codec_SDK_12.2.72/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/
```

> -- CUDA: Automatic detection of CUDA generation failed. Going to build for all known architectures

好像不用安装，直接复制文件即可

>  因为使用的是CUDA10.2，所以选择Video_Codec_SDK_10.0.26进行安装。

这样看来ffnvcodec和vulkan这两个软件也不需要安装了

参考网站:

["OpenCV is not able to find/configure CUDA SDK (required by WITH_CUDA)" when building CV4 - Jetson & Embedded Systems / Jetson Xavier NX - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/opencv-is-not-able-to-find-configure-cuda-sdk-required-by-with-cuda-when-building-cv4/147870)

#### 安装cmake

之所以重新安装cmake，是因为cmake无法检测出系统中的gpu架构

怀疑cmake版本与cuda版本不兼容

使用的cmake是通过apt-get安装的，版本号为3.22

可以通过编译安装，也可以直接下载文件后配置环境变量，也可以执行脚本文件

执行脚本文件最简单，以下是安装脚本

```sh

# 这个是预编译的二进制文件 直接解压后添加到环境变量即可
# !wget https://cmake.org/files/LatestRelease/cmake-3.30.0-linux-x86_64.tar.gz
# 这些安装太麻烦 
# !wget https://cmake.org/files/LatestRelease/cmake-3.30.0.tar.gz
# !tar -xzvf cmake-3.30.0.tar.gz cmake-3.30.0
# # 从 CMake 3.13 开始，CMake 官方推荐使用 cmake 命令而不是 bootstrap
# !cd cmake-3.30.0 && mkdir build && cd build && cmake .. && make && sudo make install
# 直接可使用
# 版本太新了
# !wget https://cmake.org/files/LatestRelease/cmake-3.30.0-linux-x86_64.sh
# !sudo bash ./cmake-3.30.0-linux-x86_64.sh --prefix=/usr/local --skip-license
# 不知道cuda与cmake的版本是否匹配
!wget https://cmake.org/files/v3.22/cmake-3.22.1-linux-x86_64.sh
!sudo bash ./cmake-3.22.1-linux-x86_64.sh --prefix=/usr/local --skip-license    
!cmake --version
```

参考网站：[Linux 安装最新版本的 CMake - RioTian - 博客园 (cnblogs.com)](https://www.cnblogs.com/RioTian/p/17979486)

cmake下载网站：[Index of /files (cmake.org)](https://cmake.org/files/)

#### apt-get安装各种依赖

1.

目前安装的依赖如下代码所示

```python
import os
import subprocess
# 运行 apt-get 更新和安装命令
subprocess.run(['sudo', 'apt-get', 'update', '-y'], check=True)
subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'], check=True)

# 安装必要的库
required_packages = [
    'ccache', 'build-essential', 'libgtk-3-dev', 'libgtk2.0-dev', 'libceres-dev', 
    'libogre-1.9-dev', 'libavutil-dev', 'libavcodec-dev', 'libavformat-dev',
    'libglib2.0-dev', 'libgtkglext1-dev', 'libavcodec-dev',
    'libavformat-dev', 'libswscale-dev', 'libavutil-dev', 'libavcodec-extra',
    'libgstreamer1.0-dev', 'libgstreamer-plugins-base1.0-dev', 'libgstreamer-plugins-good1.0-dev',
    'libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libva-dev', 'libopenblas-dev', 'libatlas-base-dev',
    'libv4l-dev', 'libxvidcore-dev', 'libx264-dev', 'libblas-dev', 'liblapack-dev',
    'gfortran', 'python3-dev', 'libeigen3-dev', 'tesseract-ocr', 'libtesseract-dev',
    'libogre-1.9-dev', 'libgflags-dev', 'libprotobuf-dev', 'protobuf-compiler',
    'libgoogle-glog-dev', 'libgtk-3-dev', 'libwnck-3-dev', 'libgnome-menu-3-dev',
    'libupower-glib-dev', 'gobject-introspection', 'libglib3.0-cil-dev', 'libgtk3.0-cil-dev',
    'automake', 'libtool', 'gtk-doc-tools', 'libgtkglext1', 'libgtkglext1-dev',
    #ubuntu 22.04不能使用
    # ubuntu22.04换源后可以使用
    'libdc1394-22', 'libdc1394-22-dev'
]

subprocess.run(['sudo', 'apt-get', 'install', '-y'] + required_packages, check=True)
```

#### 在cmake中的各种路径

注意：**行尾反斜杠**：每个选项的末尾有一个反斜杠 `\`，确保它们是命令行继续符，不要在它们后面加注释，这样可以避免命令解析问题。

**1.cuda路径配置**

```bash
# cuda的配置
'-DWITH_CUDA=ON', '-DWITH_CUDNN=ON', '-DOPENCV_DNN_CUDA=ON',
'-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda',
'-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc', # cuda工具链路径
'-DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so',
'-DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/',
```

**2.gpu架构配置**

> -- CUDA: Automatic detection of CUDA generation failed. Going to build for all known architectures
> -- Err: nvcc fatal   : Unsupported gpu architecture 'compute_20'
> -- Err: nvcc fatal   : Unsupported gpu architecture 'compute_30'
> -- Err: nvcc fatal   : Unsupported gpu architecture 'compute_35'
> -- Err: nvcc fatal   : Unsupported gpu architecture 'compute_37'

在配置安装opencv时，cmake编译gpu架构出错，同时还报以下错误

```sh
# /usr/include/c++/11/cmath:45:15: fatal error: math.h: No such file or directory
!sudo apt-get install libc6-dev -y
```

但是这样安装后系统仍然报错。

```bash
'-DCMAKE_CUDA_ARCHITECTURES=60', # gpu Tesla P100 架构代码为60 
'-DCUDA_ARCHITECTURES="60"', 
'-DCUDA_ARCH_BIN=6.0', '-DCUDA_ARCH_PTX=6.0', # 目标架构的二进制代码（BIN）和PTX代码的版本
'-DOPENCV_CMAKE_CUDA_DEBUG=1', # 开启cmake对cuda的调试
```

**3.不存在的地址`/usr/lib/include`**

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/include"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

首先发现`--   Found gtk+-2.0, version 2.24.33`，即已经安装，但是有警告

> ```bash
> CMake Warning at cmake/OpenCVUtils.cmake:885 (message):
>   ocv_check_modules(GTHREAD): can't find library 'gthread-2.0'.  Specify
>   'pkgcfg_lib_GTHREAD_gthread-2.0' manually
> Call Stack (most recent call first):
>   modules/highgui/cmake/detect_gtk.cmake:23 (ocv_check_modules)
>   modules/highgui/cmake/init.cmake:35 (include)
>   modules/highgui/cmake/init.cmake:39 (add_backend)
>   cmake/OpenCVModule.cmake:298 (include)
>   cmake/OpenCVModule.cmake:361 (_add_modules_1)
>   cmake/OpenCVModule.cmake:408 (ocv_glob_modules)
>   CMakeLists.txt:1076 (ocv_register_modules)
> ```

由于是警告，所以先别管，先解决错误

错误表明在构建 OpenCV 时，CMake 试图导入 GTK2 的相关路径 `/usr/lib/include`，但该路径并不存在。

先确定GTK2 的头文件路径是否在 `/usr/include/`

```python
import subprocess
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include'], check=True)
# ln -sf 强制覆盖已存在的链接文件
subprocess.run(['sudo', 'ln', '-sf', '/usr/include', '/usr/lib/include'], check=True)
```

使用这个暂时可以解决，但是不能解决这个路径下的文件夹的链接问题，比如以下报错

**4.不存在的地址`/usr/lib/include/freetype2`**

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/include/freetype2"
> ```

使用以下命令好像可以解决

```bash
!sudo apt-get update -y
!sudo apt-get install libfreetype6-dev -y
```

**5.不存在的地址`/usr/lib/include/gtk-2.0`**

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/include/gtk-2.0"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

使用以下代码

```python
import subprocess
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/gtk-2.0'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/include/gtk-2.0', '/usr/lib/include/gtk-2.0'], check=True)
```

**6.不存在的地址`/usr/lib/lib/x86_64-linux-gnu/gtk-2.0/include`**

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/lib/x86_64-linux-gnu/gtk-2.0/include"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

但是，添加的 `-DCMAKE_INCLUDE_PATH=/usr/include` 是告诉 CMake 将 `/usr/include` 作为包含路径，但这可能并不能直接解决 GTK2 相关的路径问题，因为 GTK2 的头文件通常位于 `/usr/include/gtk-2.0` 下，而不是直接在 `/usr/include` 下

```python
import subprocess
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/lib/x86_64-linux-gnu/gtk-2.0/include'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/lib/x86_64-linux-gnu/gtk-2.0/include', '/usr/lib/lib/x86_64-linux-gnu/gtk-2.0/include'], check=True)
```

**7.不存在的地址`/usr/lib/include/pango-1.0`**

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/include/pango-1.0"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

使用以下代码

```python
import subprocess
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/pango-1.0'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/include/pango-1.0', '/usr/lib/include/pango-1.0'], check=True)
```

**8.不存在的地址`/usr/lib/include/atk-1.0`**

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/include/atk-1.0"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

可使用以下代码

```python
import subprocess
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/atk-1.0'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/include/atk-1.0', '/usr/lib/include/atk-1.0'], check=True)
```

**9.创建文件夹`/usr/lib/include`的符号链接**

由于有太多`/usr/lib/include`下的文件夹的符号链接需要创建

使用以下代码

```python
import subprocess
# 删除现有的符号链接（如果存在）
subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/include'], check=True)

# 创建新的符号链接
subprocess.run(['sudo', 'ln', '-s', '/usr/include', '/usr/lib/include'], check=True)

# 检查符号链接
result = subprocess.run(['readlink', '/usr/lib/include'], capture_output=True, text=True)
print(f"/usr/lib/include points to: {result.stdout.strip()}")
```



**10.创建文件夹`/usr/lib/lib`的符号链接**

为了防止出现像`/usr/lib/include`那样类似的事

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/usr/lib/lib/x86_64-linux-gnu/gtk-2.0/include"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

使用以下代码

```python
import subprocess
# 删除现有的符号链接或目录（如果存在）
subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/lib'], check=True)
# 创建新的符号链接
subprocess.run(['sudo', 'ln', '-s', '/usr/lib', '/usr/lib/lib'], check=True)

# 检查符号链接的目标路径
result = subprocess.run(['readlink', '-f', '/usr/lib/lib'], capture_output=True, text=True)
print(f"/usr/lib/lib points to: {result.stdout.strip()}")
```

**11.编译时未能找到 `stdlib.h`**

> ```bash
> In file included from /usr/include/c++/9/ext/string_conversions.h:41,
>                  from /usr/include/c++/9/bits/basic_string.h:6496,
>                  from /usr/include/c++/9/string:55,
>                  from /usr/include/c++/9/bits/locale_classes.h:40,
>                  from /usr/include/c++/9/bits/ios_base.h:41,
>                  from /usr/include/c++/9/ios:42,
>                  from /usr/include/c++/9/ostream:38,
>                  from /usr/include/c++/9/iostream:39,
>                  from /kaggle/working/opencv_build/opencv/3rdparty/openexr/Half/half.h:89,
>                  from /kaggle/working/opencv_build/opencv/3rdparty/openexr/Half/half.cpp:48:
> /usr/include/c++/9/cstdlib:75:15: fatal error: stdlib.h: No such file or directory
>    75 | #include_next <stdlib.h>
>       |               ^~~~~~~~~~
> compilation terminated.
> make[2]: *** [3rdparty/openexr/CMakeFiles/IlmImf.dir/build.make:63: 3rdparty/openexr/CMakeFiles/IlmImf.dir/Half/half.cpp.o] Error 1
> make[1]: *** [CMakeFiles/Makefile2:3367: 3rdparty/openexr/CMakeFiles/IlmImf.dir/all] Error 2
> make[1]: *** Waiting for unfinished jobs....
> ```



```
```

12.

> ```bash
> !echo | gcc -E -Wp,-v -
> --------------------------------------------------------------------------------
> ignoring nonexistent directory "/usr/local/include/x86_64-linux-gnu"
> ignoring nonexistent directory "/usr/lib/gcc/x86_64-linux-gnu/9/include-fixed"
> ignoring nonexistent directory "/usr/lib/gcc/x86_64-linux-gnu/9/../../../../x86_64-linux-gnu/include"
> #include "..." search starts here:
> #include <...> search starts here:
>  /usr/lib/gcc/x86_64-linux-gnu/9/include
>  /usr/local/include
>  /usr/include/x86_64-linux-gnu
>  /usr/include
> End of search list.
> # 1 "<stdin>"
> # 1 "<built-in>"
> # 1 "<command-line>"
> # 31 "<command-line>"
> # 1 "/usr/include/stdc-predef.h" 1 3 4
> # 32 "<command-line>" 2
> # 1 "<stdin>"
> ```

输出中有一些“忽略不存在的目录”提示，尽管这些目录可能不是必需的，但如果某些配置文件或者环境依赖这些路径，这可能会引发错误

创建以下这些目录

```bash
!sudo mkdir -p /usr/local/include/x86_64-linux-gnu
!sudo mkdir -p /usr/lib/gcc/x86_64-linux-gnu/9/include-fixed
!sudo mkdir -p /usr/lib/x86_64-linux-gnu/include
!sudo mkdir -p /usr/lib/gcc/x86_64-linux-gnu/9/../../../../x86_64-linux-gnu/include
```



**13.环境变量重复**

> ```bash
> !g++ -E -x c++ - -v < /dev/null
> 
> ------------------------------------
> ignoring duplicate directory "/usr/include/"
> ignoring duplicate directory "/usr/include/"
> ignoring duplicate directory "/usr/include/"
> ignoring duplicate directory "/usr/include/"
> ignoring duplicate directory "/usr/include/x86_64-linux-gnu/c++/9"
> ignoring nonexistent directory "/usr/local/include/x86_64-linux-gnu"
> ignoring nonexistent directory "/usr/lib/gcc/x86_64-linux-gnu/9/include-fixed"
> ignoring nonexistent directory "/usr/lib/gcc/x86_64-linux-gnu/9/../../../../x86_64-linux-gnu/include"
> ignoring duplicate directory "/usr/include"
> #include "..." search starts here:
> #include <...> search starts here:
>  /usr/include/
>  .
>  /usr/include/c++/9
>  /usr/include/x86_64-linux-gnu/c++/9
>  /usr/include/c++/9/backward
>  /usr/lib/gcc/x86_64-linux-gnu/9/include
>  /usr/local/include
>  /usr/include/x86_64-linux-gnu
> End of search list.
> # 1 "<stdin>"
> # 1 "<built-in>"
> # 1 "<command-line>"
> # 1 "/usr/include/stdc-predef.h" 1 3
> # 1 "<command-line>" 2
> # 1 "<stdin>"
> ```

使用以下代码

```python
import os

def remove_duplicates(path):
    # 将路径分隔符冒号分隔的字符串分割成列表
    dirs = path.split(':')
    # 去除重复的目录
    unique_dirs = list(dict.fromkeys(dirs))
    # 将去重后的目录列表重新连接成字符串
    return ':'.join(unique_dirs)
import os
import subprocess

# 获取当前的 CPLUS_INCLUDE_PATH
cplus_include_path = os.environ.get('CPLUS_INCLUDE_PATH', '')
# 去除重复目录
updated_path = remove_duplicates(cplus_include_path)
# 更新环境变量
os.environ['CPLUS_INCLUDE_PATH'] = updated_path
# 输出更新后的 CPLUS_INCLUDE_PATH
print(f"Updated CPLUS_INCLUDE_PATH: {os.environ['CPLUS_INCLUDE_PATH']}")
```

14.不使用环境变量

因为之前发现环境变量重复，当不使用环境变量时

> ```bash
> /usr/bin/cmake: /opt/conda/lib/libcurl.so.4: no version information available (required by /usr/bin/cmake)
> ```

> ```bash
> CMake Error in modules/highgui/CMakeLists.txt:
>   Imported target "ocv.3rdparty.gtk2" includes non-existent path
> 
>     "/opt/conda/include/glib-2.0"
> 
>   in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:
> 
>   * The path was deleted, renamed, or moved to another location.
> 
>   * An install or uninstall procedure did not complete successfully.
> 
>   * The installation package was faulty and references files it does not
>   provide.
> ```

15.配置`-DWITH_ADE=OFF`

```bash
# # Scanning dependencies of target ade
!git clone https://github.com/opencv/ade.git
!cd ade && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
# In file included from /usr/include/c++/9/bits/stl_algo.h:59,
#                  from /usr/include/c++/9/algorithm:62,
#                  from /kaggle/working/ade/sources/ade/source/alloc.cpp:12:
# /usr/include/c++/9/cstdlib:75:15: fatal error: stdlib.h: No such file or directory
#    75 | #include_next <stdlib.h>
#       |               ^~~~~~~~~~
# compilation terminated.
# make[2]: *** [sources/ade/CMakeFiles/ade.dir/build.make:63: sources/ade/CMakeFiles/ade.dir/source/alloc.cpp.o] Error 1

# 在cmake中配置-DWITH_ADE=OFF
```

16.cmake命令中间有注释会报错

注释掉-DCUDA_GENERATION=Auto 自动检测的架构

cmake中的代码配置为

```bash
cmake \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_QT=OFF \
    # -DCUDA_GENERATION=Auto
    -DCUDA_GENERATION=Major6 \
    ...
    ..
```

会报错

17.配置`-DCUDA_GENERATION=Major6`

> ```bash
> CMake Error at cmake/OpenCVDetectCUDAUtils.cmake:123 (message):
>   ERROR: Maxwell, Pascal, Volta, Turing, Ampere, Lovelace, Hopper, Auto
>   Generations are supported.
> Call Stack (most recent call first):
>   cmake/OpenCVDetectCUDAUtils.cmake:229 (ocv_initialize_nvidia_device_generations)
>   cmake/OpenCVDetectCUDA.cmake:76 (ocv_set_cuda_arch_bin_and_ptx)
>   cmake/OpenCVFindLibsPerf.cmake:46 (include)
>   CMakeLists.txt:830 (include)
> ```

配置为

```bash
cmake \
    -DBUILD_EXAMPLES=OFF \
    -DWITH_QT=OFF \
    -DCUDA_GENERATION=Pascal \
    ...
    ..
```

输出正常

```bash
-- CUDA: NVCC target flags -gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-D_FORCE_INLINES
```

**18.配置python绑定相关**

```bash
cmake \
	...
	-DBUILD_opencv_python2=OFF \
    -DBUILD_NEW_PYTHON_SUPPORT=ON \
    -DBUILD_opencv_python3=ON \
    -DHAVE_opencv_python3=ON \
    -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON_DEFAULT_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON3_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON3_INCLUDE_DIR=/opt/conda/envs/newCondaEnvironment/include/python3.8 \
    -DPYTHON3_LIBRARY=/opt/conda/envs/newCondaEnvironment/lib/libpython3.8.so \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages/numpy/core/include \
    -DPYTHON3_PACKAGES_PATH=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages \
    ...
    ..
```

> ```
> --     Limited API:                 NO
> ```

绑定python

```
-DOPENCV_ENABLE_PYTHON=ON
-DOPENCV_PYTHON3_LIMITED_API=ON
```

配置成python3.10

```shell
cmake \
	...	
	-DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON_DEFAULT_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON3_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON3_INCLUDE_DIR=/opt/conda/include/python3.10 \
    -DPYTHON3_LIBRARY=/opt/conda/lib/libpython3.10.so \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/lib/python3.10/site-packages/numpy/core/include \
    -DPYTHON3_PACKAGES_PATH=/opt/conda/lib/python3.10/site-packages \
    ...
    ..
```



19.启用深度学习相关

```
-DBUILD_opencv_dnn=ON \
```

20.指定

> ```
> ```
>
> 

```
-DOpenBLAS_DIR=/usr/lib/x86_64-linux-gnu/openblas-pthread \
```

21.

> ```bash
> -- Module opencv_ovis disabled because OGRE3D was not found
> -- Checking SFM glog/gflags deps... FALSE
> -- Module opencv_sfm disabled because the following dependencies are not found: Glog/Gflags
> ```
>
> 

`/lib/x86_64-linux-gnu`加入`LD_LIBRARY_PATH`

> ```bash
> !ldconfig -p | grep atlas
> liblapack_atlas.so.3 (libc6,x86-64) => /lib/x86_64-linux-gnu/liblapack_atlas.so.3
> 	liblapack_atlas.so (libc6,x86-64) => /lib/x86_64-linux-gnu/liblapack_atlas.so
> 	libatlas.so.3 (libc6,x86-64) => /lib/x86_64-linux-gnu/libatlas.so.3
> 	libatlas.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libatlas.so
> ```

> ```bash
> !ldconfig -p | grep glog
> libglog.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libglog.so.0
> 	libglog.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libglog.so
> ```

> ```bash
> !ldconfig -p | grep gflags
> libgflags_nothreads.so.2.2 (libc6,x86-64) => /lib/x86_64-linux-gnu/libgflags_nothreads.so.2.2
> 	libgflags_nothreads.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libgflags_nothreads.so
> 	libgflags.so.2.2 (libc6,x86-64) => /lib/x86_64-linux-gnu/libgflags.so.2.2
> 	libgflags.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libgflags.so
> ```

> ```bash
> !ldconfig -p | grep lapack
> liblapack_atlas.so.3 (libc6,x86-64) => /lib/x86_64-linux-gnu/liblapack_atlas.so.3
> 	liblapack_atlas.so (libc6,x86-64) => /lib/x86_64-linux-gnu/liblapack_atlas.so
> 	liblapack.so.3 (libc6,x86-64) => /lib/x86_64-linux-gnu/liblapack.so.3
> 	liblapack.so (libc6,x86-64) => /lib/x86_64-linux-gnu/liblapack.so
> ```

> ```bash
> !ldconfig -p | grep openblas
> 
> libopenblas.so.0 (libc6,x86-64) => /lib/x86_64-linux-gnu/libopenblas.so.0
> 	libopenblas.so (libc6,x86-64) => /lib/x86_64-linux-gnu/libopenblas.so
> ```
>
> 

22.

```
-DOGRE_INCLUDE_DIR=/usr/lib/include/OGRE \
    -DOGRE_LIBRARY=/usr/lib/lib/x86_64-linux-gnu/libOgreMain.so
```

23.`/opt/conda/lib/`加入`LD_LIBRARY_PATH`



加入后开始报错

```
```

`/opt/conda/lib`本身就在`LD_LIBRARY_PATH`中，只是就在最后







```bash
# 运行 CMake 配置和安装命令
cmake_command = [
    'cmake', '-DBUILD_EXAMPLES=OFF', '-DOPENCV_PYTHON3_VERSION=3.8',
    '-DPYTHON_EXECUTABLE=/opt/conda/bin/python3',
    '-DPYTHON_DEFAULT_EXECUTABLE=/opt/conda/bin/python3',
    '-DPYTHON3_EXECUTABLE=/opt/conda/bin/python3',
    '-DPYTHON3_INCLUDE_DIR=/opt/conda/envs/newCondaEnvironment/include/python3.8',
    '-DPYTHON3_LIBRARY=/opt/conda/envs/newCondaEnvironment/lib/libpython3.8.so',
    '-DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages/numpy/core/include',
    '-DPYTHON3_PACKAGES_PATH=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages',
    '-DINSTALL_PYTHON_EXAMPLES=ON', '-DINSTALL_C_EXAMPLES=OFF', '-DWITH_QT=OFF',
    '-DCUDA_GENERATION=Auto', '-DBUILD_opencv_hdf=OFF', '-DBUILD_PERF_TESTS=OFF',
    '-DBUILD_TESTS=OFF', '-DCMAKE_BUILD_TYPE=RELEASE', '-DEIGEN_INCLUDE_PATH=/usr/include/eigen3',
    '-DProtobuf_INCLUDE_DIR=/usr/include/google/protobuf/',
    '-DProtobuf_LIBRARIES=/usr/lib/x86_64-linux-gnu/libprotobuf.so',
    '-DGLOG_INCLUDE_DIR=/usr/include/glog/',
    '-DGLOG_LIBRARY=/usr/lib/x86_64-linux-gnu/libglog.so',
    '-DGFLAGS_INCLUDE_DIR=/usr/include/gflags/',
    '-DGFLAGS_LIBRARY=/usr/lib/x86_64-linux-gnu/libgflags.so',
    '-DOGRE_DIR=/usr/include/OGRE/', '-DBUILD_opencv_cnn_3dobj=OFF',
    '-DBUILD_opencv_dnn=ON', '-DBUILD_opencv_datasets=OFF',
    '-DBUILD_opencv_aruco=OFF', '-DBUILD_opencv_tracking=OFF',
    '-DBUILD_opencv_text=OFF', '-DBUILD_opencv_stereo=OFF',
    '-DBUILD_opencv_saliency=OFF', '-DBUILD_opencv_rgbd=OFF',
    '-DBUILD_opencv_reg=OFF', '-DBUILD_opencv_ovis=OFF',
    '-DBUILD_opencv_matlab=OFF', '-DBUILD_opencv_freetype=OFF',
    '-DBUILD_opencv_dpm=OFF', '-DBUILD_opencv_face=OFF',
    '-DBUILD_opencv_dnn_superres=OFF', '-DBUILD_opencv_dnn_objdetect=OFF',
    '-DBUILD_opencv_bgsegm=OFF', '-DBUILD_opencv_cvv=OFF',
    '-DBUILD_opencv_ccalib=OFF', '-DBUILD_opencv_bioinspired=OFF',
    '-DBUILD_opencv_dnn_modern=OFF', '-DBUILD_opencv_dnns_easily_fooled=OFF',
    '-DBUILD_JAVA=OFF', '-DBUILD_opencv_python2=OFF', '-DBUILD_NEW_PYTHON_SUPPORT=ON',
    '-DBUILD_opencv_python3=OFF', '-DHAVE_opencv_python3=OFF',
    '-DWITH_OPENGL=OFF', '-DWITH_VTK=OFF', '-DFORCE_VTK=OFF', '-DWITH_TBB=ON',
    '-DWITH_GDAL=ON', '-DCUDA_FAST_MATH=ON', '-DWITH_CUBLAS=ON',
    '-DWITH_MKL=ON', '-DMKL_USE_MULTITHREAD=ON', '-DOPENCV_ENABLE_NONFREE=ON',
    '-DWITH_CUDA=ON', '-DWITH_CUDNN=ON', '-DOPENCV_DNN_CUDA=ON',
    '-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda',
    '-DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so',
    '-DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/',
    '-DNVCC_FLAGS_EXTRA=--default-stream per-thread', '-DWITH_NVCUVID=OFF',
    '-DBUILD_opencv_cudacodec=OFF', '-DMKL_WITH_TBB=ON', '-DWITH_FFMPEG=ON',
    '-DMKL_WITH_OPENMP=ON', '-DWITH_XINE=ON', '-DENABLE_PRECOMPILED_HEADERS=OFF',
    '-DWITH_AVRESAMPLE=OFF',
    '-DGLIB_INCLUDE_DIR=/usr/include/glib-2.0',
    '-DGTK_INCLUDE_DIR=/usr/include/gtk-3.0',
    '-DGSTREAMER_INCLUDE_DIR=/usr/include/gstreamer-1.0',
    '-Ddc1394_DIR=/usr/include/libdc1394-2',
    '-DWITH_JULIA=ON',
    '-DJULIA_EXECUTABLE=/usr/local/julia-1.7.3/bin/julia',
    '-DJULIA_INCLUDE_DIR=/usr/local/julia-1.7.3/include/julia',
    '-DJULIA_LIBRARIES=/usr/local/julia-1.7.3/lib/libjulia.so',
    '-DVA_INCLUDE_DIR=/usr/include/va',
    '-DOpenBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu',
    '-DOpenBLAS_LIB=/usr/lib/x86_64-linux-gnu/libopenblas.so',
    '-DAtlas_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/atlas',
    '-DAtlas_LIBRARIES=/usr/lib/x86_64-linux-gnu/libatlas.so',
    '-DAtlas_CLAPACK_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/atlas',
    '-DAtlas_CLAPACK_LIBRARY=/usr/lib/x86_64-linux-gnu/libatlas.so',
    '-DOGRE_INCLUDE_DIR=/usr/include/OGRE',
    '-DOGRE_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu/OGRE-1.9.0',
    '-DLAPACK_INCLUDE_DIR=/usr/include/x86_64-linux-gnu',
    '-DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so',
    '-DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so',
    '-DOPENCV_DNN=ON',
    '-DCMAKE_INCLUDE_PATH=/usr/include',
    '-DCMAKE_INSTALL_PREFIX=/usr/local/', '-DOPENCV_GENERATE_PKGCONFIG=ON',
    '-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules', '..'
]
```

#### 配置软链接

在安装时，有一些软件即使配置了位置仍然识别不了软件，因此进行软链接的创建

具体的创建代码如下

```python
# 创建软链接
# CMake Error in modules/highgui/CMakeLists.txt:
#   Imported target "ocv.3rdparty.gtk3" includes non-existent path

#     "/usr/lib/include/gtk-3.0"

# subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/gstreamer-1.0'], check=True)
# subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/gtk-3.0'], check=True)

# subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/include'], check=True)
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include'], check=True)
# ln -sf 强制覆盖已存在的链接文件
subprocess.run(['sudo', 'ln', '-sf', '/usr/include', '/usr/lib/include'], check=True)

# subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/include/x86_64-linux-gnu'], check=True)
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/x86_64-linux-gnu'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/include/x86_64-linux-gnu', '/usr/lib/include/x86_64-linux-gnu'], check=True)

# subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/lib/x86_64-linux-gnu/glib-2.0/include'], check=True)
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/lib/x86_64-linux-gnu/glib-2.0/include'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/lib/x86_64-linux-gnu/glib-2.0/include', '/usr/lib/lib/x86_64-linux-gnu/glib-2.0/include'], check=True)

# subprocess.run(['sudo', 'ln', '-sf', '/usr/lib/x86_64-linux-gnu/libdc1394.so', '/usr/lib/libdc1394.so'], check=True)
```

#### 配置环境变量

为了进一步防止找不到软件路径，进行环境变量的配置

具体的配置代码如下

```python
os.environ['LD_LIBRARY_PATH'] = "/usr/local/lib:/usr/lib/x86_64-linux-gnu:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PKG_CONFIG_PATH'] = "/usr/lib/x86_64-linux-gnu/pkgconfig:" + \
                                "/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')
os.environ['INCLUDE_PATH'] = '/usr/local/include:/usr/include' + os.environ.get('INCLUDE_PATH', '')

os.environ['CPLUS_INCLUDE_PATH']="/usr/include/:" + os.environ.get('CPLUS_INCLUDE_PATH', '')
```

#### 安装opencv

从源码编译安装opencv需要下载opencv和opencv_contrib两个软件包

直接使用aptget安装的opencv版本太低，不支持项目

具体下载编译安装的代码如下：

```python
# 安装opencv
# 

import os
import subprocess

# 设置环境变量
os.environ['PKG_CONFIG_PATH'] = "/usr/lib/x86_64-linux-gnu/pkgconfig:" + \
                                "/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')

# 运行 apt-get 更新和安装命令
subprocess.run(['sudo', 'apt-get', 'update', '-y'], check=True)
subprocess.run(['sudo', 'apt-get', 'upgrade', '-y'], check=True)

# 安装必要的库
required_packages = [
    'ccache', 'build-essential', 'libgtk-3-dev', 'libgtk2.0-dev', 'libceres-dev', 
    'libogre-1.9-dev', 'libavutil-dev', 'libavcodec-dev', 'libavformat-dev',
    'libglib2.0-dev', 'libgtkglext1-dev', 'libavcodec-dev',
    'libavformat-dev', 'libswscale-dev', 'libavutil-dev', 'libavcodec-extra',
    'libgstreamer1.0-dev', 'libgstreamer-plugins-base1.0-dev', 'libgstreamer-plugins-good1.0-dev',
    'libjpeg-dev', 'libpng-dev', 'libtiff-dev', 'libva-dev', 'libopenblas-dev', 'libatlas-base-dev',
    'libv4l-dev', 'libxvidcore-dev', 'libx264-dev', 'libblas-dev', 'liblapack-dev',
    'gfortran', 'python3-dev', 'libeigen3-dev', 'tesseract-ocr', 'libtesseract-dev',
    'libogre-1.9-dev', 'libgflags-dev', 'libprotobuf-dev', 'protobuf-compiler',
    'libgoogle-glog-dev', 'libgtk-3-dev', 'libwnck-3-dev', 'libgnome-menu-3-dev',
    'libupower-glib-dev', 'gobject-introspection', 'libglib3.0-cil-dev', 'libgtk3.0-cil-dev',
    'automake', 'libtool', 'gtk-doc-tools', 'libgtkglext1', 'libgtkglext1-dev',
    #ubuntu 22.04不能使用
    # ubuntu22.04换源后可以使用
    'libdc1394-22', 'libdc1394-22-dev'
]

subprocess.run(['sudo', 'apt-get', 'install', '-y'] + required_packages, check=True)
#!sudo ldconfig
subprocess.run(['sudo', 'ldconfig'], check=True)



# 创建软链接
# CMake Error in modules/highgui/CMakeLists.txt:
#   Imported target "ocv.3rdparty.gtk3" includes non-existent path

#     "/usr/lib/include/gtk-3.0"

# subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/gstreamer-1.0'], check=True)
# subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/gtk-3.0'], check=True)

# subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/include'], check=True)
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include'], check=True)
# ln -sf 强制覆盖已存在的链接文件
subprocess.run(['sudo', 'ln', '-sf', '/usr/include', '/usr/lib/include'], check=True)

# subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/include/x86_64-linux-gnu'], check=True)
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/include/x86_64-linux-gnu'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/include/x86_64-linux-gnu', '/usr/lib/include/x86_64-linux-gnu'], check=True)

# subprocess.run(['sudo', 'rm', '-rf', '/usr/lib/lib/x86_64-linux-gnu/glib-2.0/include'], check=True)
subprocess.run(['sudo', 'mkdir', '-p', '/usr/lib/lib/x86_64-linux-gnu/glib-2.0/include'], check=True)
subprocess.run(['sudo', 'ln', '-sf', '/usr/lib/x86_64-linux-gnu/glib-2.0/include', '/usr/lib/lib/x86_64-linux-gnu/glib-2.0/include'], check=True)

# subprocess.run(['sudo', 'ln', '-s', '/usr/lib/x86_64-linux-gnu/libdc1394.so', '/usr/lib/libdc1394.so'], check=True)


#!sudo ldconfig
subprocess.run(['sudo', 'ldconfig'], check=True)

os.chdir('/kaggle/working/')  

# 克隆 OpenCV 源代码
os.makedirs('./opencv_build', exist_ok=True)
subprocess.run(['git', 'clone', 'https://github.com/opencv/opencv.git'], cwd='./opencv_build', check=True)
subprocess.run(['git', 'clone', 'https://github.com/opencv/opencv_contrib.git'], cwd='./opencv_build', check=True)

# 检出特定版本
subprocess.run(['git', 'checkout', '4.10.0'], cwd='./opencv_build/opencv', check=True)
subprocess.run(['git', 'checkout', '4.10.0'], cwd='./opencv_build/opencv_contrib', check=True)

# 创建用于编译的临时目录
os.makedirs('./opencv_build/opencv/build', exist_ok=True)

# 运行 CMake 配置和安装命令
cmake_command = [
    'cmake', '-DBUILD_EXAMPLES=OFF', '-DOPENCV_PYTHON3_VERSION=3.8',
    '-DPYTHON_EXECUTABLE=/opt/conda/bin/python3',
    '-DPYTHON_DEFAULT_EXECUTABLE=/opt/conda/bin/python3',
    '-DPYTHON3_EXECUTABLE=/opt/conda/bin/python3',
    '-DPYTHON3_INCLUDE_DIR=/opt/conda/envs/newCondaEnvironment/include/python3.8',
    '-DPYTHON3_LIBRARY=/opt/conda/envs/newCondaEnvironment/lib/libpython3.8.so',
    '-DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages/numpy/core/include',
    '-DPYTHON3_PACKAGES_PATH=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages',
    '-DINSTALL_PYTHON_EXAMPLES=ON', '-DINSTALL_C_EXAMPLES=OFF', '-DWITH_QT=OFF',
    '-DCUDA_GENERATION=Auto', '-DBUILD_opencv_hdf=OFF', '-DBUILD_PERF_TESTS=OFF',
    '-DBUILD_TESTS=OFF', '-DCMAKE_BUILD_TYPE=RELEASE', '-DEIGEN_INCLUDE_PATH=/usr/include/eigen3',
    '-DProtobuf_INCLUDE_DIR=/usr/include/google/protobuf/',
    '-DProtobuf_LIBRARIES=/usr/lib/x86_64-linux-gnu/libprotobuf.so',
    '-DGLOG_INCLUDE_DIR=/usr/include/glog/',
    '-DGLOG_LIBRARY=/usr/lib/x86_64-linux-gnu/libglog.so',
    '-DGFLAGS_INCLUDE_DIR=/usr/include/gflags/',
    '-DGFLAGS_LIBRARY=/usr/lib/x86_64-linux-gnu/libgflags.so',
    '-DOGRE_DIR=/usr/include/OGRE/', '-DBUILD_opencv_cnn_3dobj=OFF',
    '-DBUILD_opencv_dnn=ON', '-DBUILD_opencv_datasets=OFF',
    '-DBUILD_opencv_aruco=OFF', '-DBUILD_opencv_tracking=OFF',
    '-DBUILD_opencv_text=OFF', '-DBUILD_opencv_stereo=OFF',
    '-DBUILD_opencv_saliency=OFF', '-DBUILD_opencv_rgbd=OFF',
    '-DBUILD_opencv_reg=OFF', '-DBUILD_opencv_ovis=OFF',
    '-DBUILD_opencv_matlab=OFF', '-DBUILD_opencv_freetype=OFF',
    '-DBUILD_opencv_dpm=OFF', '-DBUILD_opencv_face=OFF',
    '-DBUILD_opencv_dnn_superres=OFF', '-DBUILD_opencv_dnn_objdetect=OFF',
    '-DBUILD_opencv_bgsegm=OFF', '-DBUILD_opencv_cvv=OFF',
    '-DBUILD_opencv_ccalib=OFF', '-DBUILD_opencv_bioinspired=OFF',
    '-DBUILD_opencv_dnn_modern=OFF', '-DBUILD_opencv_dnns_easily_fooled=OFF',
    '-DBUILD_JAVA=OFF', '-DBUILD_opencv_python2=OFF', '-DBUILD_NEW_PYTHON_SUPPORT=ON',
    '-DBUILD_opencv_python3=OFF', '-DHAVE_opencv_python3=OFF',
    '-DWITH_OPENGL=OFF', '-DWITH_VTK=OFF', '-DFORCE_VTK=OFF', '-DWITH_TBB=ON',
    '-DWITH_GDAL=ON', '-DCUDA_FAST_MATH=ON', '-DWITH_CUBLAS=ON',
    '-DWITH_MKL=ON', '-DMKL_USE_MULTITHREAD=ON', '-DOPENCV_ENABLE_NONFREE=ON',
    '-DWITH_CUDA=ON', '-DWITH_CUDNN=ON', '-DOPENCV_DNN_CUDA=ON',
    '-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda',
    '-DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so',
    '-DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/',
    '-DNVCC_FLAGS_EXTRA=--default-stream per-thread', '-DWITH_NVCUVID=OFF',
    '-DBUILD_opencv_cudacodec=OFF', '-DMKL_WITH_TBB=ON', '-DWITH_FFMPEG=ON',
    '-DMKL_WITH_OPENMP=ON', '-DWITH_XINE=ON', '-DENABLE_PRECOMPILED_HEADERS=OFF',
    '-DWITH_AVRESAMPLE=OFF',
    '-DGLIB_INCLUDE_DIR=/usr/include/glib-2.0',
    '-DGTK_INCLUDE_DIR=/usr/include/gtk-3.0',
    '-DGSTREAMER_INCLUDE_DIR=/usr/include/gstreamer-1.0',
    '-Ddc1394_DIR=/usr/include/libdc1394-2',
    '-DWITH_JULIA=ON',
    '-DJULIA_EXECUTABLE=/usr/local/julia-1.7.3/bin/julia',
    '-DJULIA_INCLUDE_DIR=/usr/local/julia-1.7.3/include/julia',
    '-DJULIA_LIBRARIES=/usr/local/julia-1.7.3/lib/libjulia.so',
    '-DVA_INCLUDE_DIR=/usr/include/va',
    '-DOpenBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu',
    '-DOpenBLAS_LIB=/usr/lib/x86_64-linux-gnu/libopenblas.so',
    '-DAtlas_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/atlas',
    '-DAtlas_LIBRARIES=/usr/lib/x86_64-linux-gnu/libatlas.so',
    '-DAtlas_CLAPACK_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/atlas',
    '-DAtlas_CLAPACK_LIBRARY=/usr/lib/x86_64-linux-gnu/libatlas.so',
    '-DOGRE_INCLUDE_DIR=/usr/include/OGRE',
    '-DOGRE_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu/OGRE-1.9.0',
    '-DLAPACK_INCLUDE_DIR=/usr/include/x86_64-linux-gnu',
    '-DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so',
    '-DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so',
    '-DOPENCV_DNN=ON',
    '-DCMAKE_INCLUDE_PATH=/usr/include',
    '-DCMAKE_INSTALL_PREFIX=/usr/local/', '-DOPENCV_GENERATE_PKGCONFIG=ON',
    '-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules', '..'
]
#!sudo ldconfig
subprocess.run(['sudo', 'ldconfig'], check=True)
# export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
# 更新 LD_LIBRARY_PATH 环境变量
os.environ['LD_LIBRARY_PATH'] = "/usr/local/lib:/usr/lib/x86_64-linux-gnu:" + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['PKG_CONFIG_PATH'] = "/usr/lib/x86_64-linux-gnu/pkgconfig:" + \
                                "/usr/lib/pkgconfig:/usr/local/lib/pkgconfig:" + os.environ.get('PKG_CONFIG_PATH', '')
os.environ['INCLUDE_PATH'] = '/usr/local/include:/usr/include' + os.environ.get('INCLUDE_PATH', '')

os.environ['CPLUS_INCLUDE_PATH']="/usr/include/:" + os.environ.get('CPLUS_INCLUDE_PATH', '')


# # 建议清理之前的 CMake 配置缓存
subprocess.run(['rm', '-rf', 'CMakeCache.txt', 'CMakeFiles/'], cwd='./opencv_build/opencv/build', check=True)


subprocess.run(cmake_command, cwd='./opencv_build/opencv/build', check=True)

# 编译并安装 OpenCV

subprocess.run(['make', '-j', str(os.cpu_count())], cwd='./opencv_build/opencv/build', check=True)
subprocess.run(['sudo', 'make', 'install'], cwd='./opencv_build/opencv/build', check=True)

# 验证 OpenCV 是否安装成功
subprocess.run(['pkg-config', '--modversion', 'opencv4'], check=True)
```

运行提示

1.GPU架构自动检测问题

> -- CUDA detected: 10.2
> -- CUDA: Automatic detection of CUDA generation failed. Going to build for all known architectures
> CMake Warning at cmake/OpenCVDetectCUDAUtils.cmake:187 (message):
>   CUDA: Autodetection arch list is empty.  Please enable
>   OPENCV_CMAKE_CUDA_DEBUG=1 and check/specify
>   OPENCV_CUDA_DETECTION_NVCC_FLAGS variable
> Call Stack (most recent call first):
>   cmake/OpenCVDetectCUDAUtils.cmake:286 (ocv_filter_available_architecture)
>   cmake/OpenCVDetectCUDA.cmake:76 (ocv_set_cuda_arch_bin_and_ptx)
>   cmake/OpenCVFindLibsPerf.cmake:46 (include)
>   CMakeLists.txt:830 (include)
>
> CMake Error at cmake/OpenCVDetectCUDAUtils.cmake:297 (list):
>   list GET given empty list
> Call Stack (most recent call first):
>   cmake/OpenCVDetectCUDA.cmake:76 (ocv_set_cuda_arch_bin_and_ptx)
>   cmake/OpenCVFindLibsPerf.cmake:46 (include)
>   CMakeLists.txt:830 (include)



2.关于libavresample

> ```bash
> -- Checking for module 'libavresample'
> --   No package 'libavresample' found
> ```

参考chatgpt提供的方法，在编译ffmpeg代码中添加

`./configure --enable-libavresample`

但是添加的有问题，configure无法识别这个选项

在新的ffmpeg中libavresample不再支持，被其他软件取代

不知道这个软件的缺失会不会影响opencv的编译和安装

3.关于tesseract

> ```bash
> --   Found tesseract, version 4.1.1
> -- Tesseract:   YES (ver 4.1.1)
> -- Can't use Tesseract (details: https://github.com/opencv/opencv_contrib/pull/2220)
> ```

（1）把gcc的版本提上来

之前安装cuda10.2时，安装了gcc-7，并且指定当前系统的gcc版本为7

在安装完cuda10.2后，应该把gcc的版本退回去，即为gcc-11

点开网站（ https://github.com/opencv/opencv_contrib/pull/2220）发现

opencv3.4要在C++11上编译

因此之前转换C++7之后需要换回C++11

（2）不使用cuda10.2

4.已经安装HDF5

> -- Found HDF5: /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so;/usr/lib/x86_64-linux-gnu/libcrypto.so;/usr/lib/x86_64-linux-gnu/libcurl.so;/usr/lib/x86_64-linux-gnu/libpthread.a;/usr/lib/x86_64-linux-gnu/libsz.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libdl.a;/usr/lib/x86_64-linux-gnu/libm.so (found version "1.10.7") 

5.关于Julia

> -- JlCxx found but not source build - disabling Julia module

> -- Module opencv_ovis disabled because OGRE3D was not found

> -- Module opencv_sfm disabled because the following dependencies are not found: Glog/Gflags



> Unavailable:  cannops cvv java julia matlab ovis python2 sfm ts viz

> -- NVCUVENC: Header not found, WITH_NVCUVENC requires Nvidia encoding library header /usr/local/cuda;/usr/local/cuda/include/nvEncodeAPI.h



参考网站：

[【环境配置篇】保姆级教学之Ubuntu20.04上编译OpenCV+CUDA_ubuntu opencv cuda-CSDN博客](https://blog.csdn.net/ChunjieShan/article/details/125391238)

但是目前opencv没有安装成功

可能的原因有

（1）系统不支持openGL，以至于Video_Codec_SDK_10.0.26无法安装成功

（2）cuda版本为10.2，与当前编译安装的软件包的版本不匹配



系统环境设置为cuda10.2时

编译opencv时会提示cuda版本过低，应该使用cuda11以上的版本

1.cuda10.2 

2.cuda12.3







### 安装denseflow

denseflow安装依赖opencv的安装，但是opencv没有安装成功

所以下面的代码没有进行测试

只是从脚本中改写

#### 安装Boost Libray

编译并安装Boost Libray的代码如下：

```bash
# 下载和编译boost

!wget https://boostorg.jfrog.io/artifactory/main/release/1.86.0/source/boost_1_86_0.tar.gz
!tar -xvzf boost_1_86_0.tar.gz boost_1_86_0
!cd boost_1_86_0 && sh ./bootstrap.sh --prefix=/usr/local && ./b2 install
!echo export BOOST_ROOT=/usr/local
```

具体的代码还需进行测试

#### 安装HDF5 Libary(Optional)

hdf5的安装是可选的，因为在安装opencv时，它就已经被安装了

因此不需要重复安装

但是可能的编译并安装HDF5 Libary的代码如下：

```bash
# 下载和编译hdf5

!wget https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_0/source/hdf5-1.12.0.tar.gz
!cd hdf5-1.12.0 && ./configure --prefix=/usr/local --enable-cxx --disable-static && make -j"$(nproc)" && make install
```

具体的代码还需进行测试

#### 安装denseflow

编译并安装denseflow的代码如下：

```bash
#下载和编译denseflow

!wget https://codeload.github.com/open-mmlab/denseflow/tar.gz/master
!tar -xvzf denseflow-master.tar.gz denseflow-master
!cd denseflow-master && mkdir -p build
!cd denseflow-master/build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && make -j"$(nproc)" && make install
```

具体的代码还需进行测试

#### 后续软件包及依赖的安装

由于opencv未安装成功，denseflow无法编译安装

之后的过程无法处理

之后具体的安装及调错的过程还需时间和测试

## 上传各种文件

### 上传项目代码

由于代码运行环境与kaggle服务器的环境不完全匹配

在后期可能需要调试，或者会出现不知名错误

因此需要对代码进行频繁的修改

于是先将本地代码上传至github，得到github项目地址

再从kaggle命令块中利用github项目地址进行项目的下载

当需要修改代码时，只需在本地使用pycharm对代码进行修改，再将修改部分传至github，再重新从github处下载新项目

### 上传数据集和模型文件

因为数据集和模型文件是不需要更改的，上传至Kaggle网站上后，再将其引入，放在/kaggle/working/文件夹下

当将数据集压缩成zip文件上传后，网站会自动解压

只需配置好数据集的地址，便可以使用

模型文件的处理相同

## 数据集处理与特征提取

源项目使用的数据集是CAS(ME)2和SAMM-LV，目前只有CAS(ME)2的数据集，因此只能进行这一个数据集的处理。

数据集处理的环境与训练模型的环境不同。

### 数据集处理

数据集处理的环境与训练模型的环境不同。

数据集处理目前遇到bug、环境不匹配，还需调试。

目前的将视频进行帧采样、剪切和记录人脸关键点与兴趣区域已经完成，但在下一步出了问题。

虽然数据集处理的过程除了问题，但不影响模型的训练。

因为源项目的作者提供处理好的数据集，包括提取的特征和标签文件。

可以直接进行运行训练部分的代码。

### 特征提取

在进行提取光流时遇到了问题

```python
import os
# os.environ['CUDA_VISIBLE_DEVICES']    = '3, 4'
# 只有0可以用
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # 对cas(me)^2而言 不需要进行sampling
# print("================ sampling ================")
# # 顶点 采样？
# apex_sampling(opt)
print("================ crop ================")
crop(opt)

print("================ record ================")
record_face_and_landmarks(opt)
print("================ optical flow ================")
optflow(opt)
print("================ feature ================")
feature(opt)
print("================ feature segment ================")
segment_for_train(opt)
segment_for_test(opt)
```

代码中`optflow(opt)`之前的都可正常运行，但是红框中的部分报错，报错内容为

```bash
sh: 1: denseflow: not found
```

报错位置为

```python
# sh: 1: denseflow: not found
# 需要安装desenflow
# 处理视频 获取光流特征
cmd = (f'denseflow "{str(type_item)}" -b=10 -a=tvl1 'f'-s={opt_step} -if -o="{new_sub_dir_path}"')
os.system(cmd)
tq.update()
```

需要安装denseflow，但是在安装densflow的过程出错，可能是版本不匹配，需要继续排查。

对数据集进行预处理和特征提取的代码为

```bash
!cd /kaggle/working/ME-GCN-Project/feature_extraction
!python /kaggle/working/ME-GCN-Project/feature_extraction/new_all.py
```

具体的调错过程仍然在继续。

## 训练和测试模型

**图片转换为npz文件**

运行代码时没有读取图片数据，但是读取的是npz文件。

参考：

[在pytorch中使用用npz文件保存的预训练模型_npz是什么文件-CSDN博客](https://blog.csdn.net/weixin_44020747/article/details/115208672)

[1张或多张jpg/png文件转换为npz文件并读取_png转npz-CSDN博客](https://blog.csdn.net/weixin_44669966/article/details/122566037)

### 训练和测试模型

由于项目的作者有现成的处理好的数据集，即从CAS(ME)2和SAMM-LV这两个数据集中提取的特征与标签统一的npz文件。

即使没有原始图像和视频也可以进行模型的训练和测试。

如果需要重新提取图像特征和标签，则需要原始数据集进行操作

因此配置好python环境和cuda环境（在之后的测试中甚至cuda12.3也可以运行）后

修改代码中数据集和模型的地址

1.在`config.yaml`文件中

修改项目地址、特征文件地址、标签文件地址

```yaml
# cas(me)^2
project_root: "/kaggle/working/ME-GCN-Project"
feature_root: ~
segment_feat_root: "/kaggle/working/ME-GCN-Project/features/cas(me)^2/feature_segment"
model_save_root: ~
output_dir_name: ~
anno_csv: "/kaggle/working/ME-GCN-Project/info_csv/cas(me)_new.csv"
num_workers: 2
device: 'cuda:0'
  
# samm
project_root: "/kaggle/working/ME-GCN-Project"
feature_root: ~
segment_feat_root: "/kaggle/working/ME-GCN-Project/features/samm_25/feature_segment_25"
model_save_root: ~
output_dir_name: ~
anno_csv: "/kaggle/working/ME-GCN-Project/info_csv/samm_new_25.csv"
num_workers: 2
device: 'cuda:0'
```

2.在`pipeline.sh`文件中

修改输出地址、训练文件地址、评估文件地址、计算总分地址

```sh
# cas(me)^2
OUTPUT="/kaggle/working/output/casme"
DATASET="cas(me)^2"

# samm
OUTPUT="/kaggle/working/output/samm"
DATASET="samm"

for i in ${SUB_LIST[@]}
do     
    echo "************ Currently running subject: ${i} ************"$'\n'
    # comment the line below if evaluating on available ckpts.
    python /kaggle/working/ME-GCN-Project/train.py --dataset $DATASET --output $OUTPUT --subject ${i}  # for training
    python /kaggle/working/ME-GCN-Project/eval.py --dataset $DATASET --output $OUTPUT --subject ${i}   # for evaluation
done

#output final metrics
python /kaggle/working/ME-GCN-Project/calc_final_score.py --output $OUTPUT
```

3.在Kaggle的命令块中下载项目，并安装项目所需要的python依赖

```python
# 下载项目代码并安装依赖
# 3~10min

# !rm ME-GCN-Project -rf
!git clone https://github.com/garlic0000/ME-GCN-Project.git
!cd ME-GCN-Project
# 安装依赖
!pip install -r /kaggle/working/ME-GCN-Project/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4.进行模型的训练和测试

```bash
# 进行模型的训练和评估
# !rm -rf output
!bash /kaggle/working/ME-GCN-Project/pipeline.sh
```

大概运行一个多小时，可以跑完100个epoch

设置完环境同时跑完两个数据集大概三个多小时

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

### 结果展示与分析

#### 在CAS(ME)2上的训练结果

1.在CAS(ME)2上的训练过程如下所示

```
************ Currently running subject: casme_015 ************

Starting training...

Using GPU: cuda:0 

[Epoch 000/100]	Loss 0.16210(train)	Current Learning rate 0.01000

weight file save in /kaggle/working/output/casme/casme_015/models/checkpoint_epoch_000.pth.tar

[Epoch 001/100]	Loss 0.05543(train)	Current Learning rate 0.00960

weight file save in /kaggle/working/output/casme/casme_015/models/checkpoint_epoch_001.pth.tar
```

2.在CAS(ME)2上的评估过程如下所示

```
Starting evaluating...

Using GPU: cuda:0 

Evaluating ckpt of [Epoch 022/100]

output_csv file save in /kaggle/working/output/casme/casme_015/output_csv/proposals_epoch_022.csv

nms_csv file save in /kaggle/working/output/casme/casme_015/nms_csv/final_proposals_epoch_022.csv

Evaluating ckpt of [Epoch 023/100]

output_csv file save in /kaggle/working/output/casme/casme_015/output_csv/proposals_epoch_023.csv

nms_csv file save in /kaggle/working/output/casme/casme_015/nms_csv/final_proposals_epoch_023.csv
```

3.在CAS(ME)2上训练和评估完成的最终结果如下所示

```
Micro result: TP:6.0, FP:34.0, FN:51.0
Precision =  0.15
Recall =  0.1053
F1-Score =  0.1237
Macro result: TP:136.0, FP:189.0, FN:164.0
Precision =  0.4185
Recall =  0.4533
F1-Score =  0.4352
Total result: TP:142.0, FP:223.0, FN:215.0
Precision =  0.389
Recall =  0.3978
F1-Score =  0.3934
```

#### 在SAMM上的训练结果

1.在SAMM上的训练过程如下所示

```
************ Currently running subject: samm_007 ************

Starting training...

Using GPU: cuda:0 

[Epoch 000/100]	Loss 0.19448(train)	Current Learning rate 0.01000

weight file save in /kaggle/working/output/samm/samm_007/models/checkpoint_epoch_000.pth.tar

[Epoch 001/100]	Loss 0.10270(train)	Current Learning rate 0.00960

weight file save in /kaggle/working/output/samm/samm_007/models/checkpoint_epoch_001.pth.tar
```

2.在SAMM上的评估过程如下所示

```
Starting evaluating...

Using GPU: cuda:0 

Evaluating ckpt of [Epoch 019/100]

Evaluating ckpt of [Epoch 020/100]

output_csv file save in /kaggle/working/output/samm/samm_007/output_csv/proposals_epoch_020.csv

nms_csv file save in /kaggle/working/output/samm/samm_007/nms_csv/final_proposals_epoch_020.csv

Evaluating ckpt of [Epoch 021/100]

output_csv file save in /kaggle/working/output/samm/samm_007/output_csv/proposals_epoch_021.csv

nms_csv file save in /kaggle/working/output/samm/samm_007/nms_csv/final_proposals_epoch_021.csv
```

3.在SAMM上训练和评估完成的最终结果如下所示

```
Micro result: TP:21.0, FP:56.0, FN:138.0
Precision =  0.2727
Recall =  0.1321
F1-Score =  0.178
Macro result: TP:175.0, FP:296.0, FN:168.0
Precision =  0.3715
Recall =  0.5102
F1-Score =  0.43
Total result: TP:196.0, FP:352.0, FN:306.0
Precision =  0.3577
Recall =  0.3904
F1-Score =  0.3733
```

#### 在两个数据集上的综合结果

两个数据集上的综合结果如表4-1所示

<center>表4-1

|   **Dataset**    |    **CAS(ME)^2**     |    **CAS(ME)^2**     | **CAS(ME)^2** |     **SAMM-LV**      |     **SAMM-LV**      | **SAMM-LV** |
| :--------------: | :------------------: | :------------------: | :-----------: | :------------------: | :------------------: | :---------: |
|  **Expression**  | **macro-expression** | **micro-expression** |  **overall**  | **macro-expression** | **micro-expression** | **overall** |
| **Total number** |         300          |          57          |      357      |         343          |         159          |     502     |
|      **TP**      |        136.0         |         6.0          |     142.0     |        175.0         |         21.0         |    196.0    |
|      **FP**      |        189.0         |         34.0         |     223.0     |        296.0         |         56.0         |    352.0    |
|      **FN**      |        164.0         |         51.0         |     215.0     |        168.0         |        138.0         |    306.0    |
|  **Precision**   |        0.4185        |        0.1500        |    0.3890     |        0.3715        |        0.2727        |   0.3577    |
|    **Recall**    |        0.4533        |        0.1053        |    0.3978     |        0.5102        |        0.1321        |   0.3904    |
|   **F1-score**   |        0.4352        |        0.1237        |    0.3934     |        0.4300        |        0.1780        |   0.3733    |

将项目复现的结果与源项目的作者训练的结果进行比较 ，结果如表4-2所示

<center>表4-2

|  **Dataset**   |    **CAS(ME)^2**     |    **CAS(ME)^2**     | **CAS(ME)^2** |     **SAMM-LV**      |     **SAMM-LV**      | **SAMM-LV** |
| :------------: | :------------------: | :------------------: | :-----------: | :------------------: | :------------------: | :---------: |
| **Expression** | **macro-expression** | **micro-expression** |  **overall**  | **macro-expression** | **micro-expression** | **overall** |
|    AUW-GCN     |        0.4235        |        0.1538        |    0.3834     |        0.4293        |        0.1984        |   0.3728    |
|   Re-AUW-GCN   |      **0.4352**      |        0.1237        |  **0.3934**   |      **0.4300**      |        0.1780        | **0.3733**  |

在数据集CAS(ME)2上和数据集SAMM-LV上，复现的结果中宏表情和综合F1分数均高于源项目的F1分数。可能的原因是CUDA版本。
源项目使用的CUDA版本是10.2，但是在Kaggle服务器中使用CUDA12.1也可以进行模型的训练，因此可能的原因是CUDA。

每次运行的结果都不一样，有时高有时低，结果不稳定，不知与什么有关？

多环境测试：

1.Ubuntu20.04 python3.8 pytorch1.13.1 cuda12.1

2.Ubuntu 22.04 python3.10 pytorch 2.4 cuda 12.3

这两个环境的运行结果差异很大，一个宏表情以之前的高，但微表情低

一个宏表情比之前低，但微表情高

是否要控制变量

## 运行问题

1.关于num workers的设置

> This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.

修改num workers的值，根据警告信息，将8修改为4

```python
# define dataset & loader
dataset = LOSO_DATASET(opt, 'test', subject)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt['batch_size'], 
                                         shuffle=False,
                                         # num_workers=8
                                         num_workers=4,
                                         pin_memory=True, 
                                         drop_last=False)
```

参考网站：

[pytorch中DataLoader的num_workers参数详解与设置大小建议](https://blog.csdn.net/qq_28057379/article/details/115427052)

2.关于pretrained参数

> UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
>
> UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.

根据警告信息，需要将形参`pretrained改为`weights

```python
# 在feature_extraction/retinaface/models/retinaface.py中第71行的代码backbone = models.resnet50(pretrained=cfg['pretrain'])

elif cfg['name'] == 'Resnet50':
    import torchvision.models as models
     # UserWarning: The parameter 'pretrained' is deprecated since 0.13
     # and may be removed in the future, please use 'weights' instead.
     # backbone = models.resnet50(pretrained=cfg['pretrain'])
     backbone = models.resnet50(weights=cfg['pretrain'])
```

参考网站：

[深度学习：UserWarning: The parameter ‘pretrained‘ is deprecated since 0.13..解决办法_userwarning: the parameter 'pretrained' is depreca-CSDN博客](https://blog.csdn.net/qudunan6468/article/details/133808253)

3.关于align_corners=True

> UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired.

在项目代码中没有发现align_corners=True相关的代码，有以下两种猜测：

（1）可能使用python软件包版本太低

（2）可能编译安装的库版本太低

还需要继续查找和调试。

参考网站：

[UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1-CSDN博客](https://blog.csdn.net/m0_51233386/article/details/128489132)

4.关于os.fork

5.关于weights_only=True



4.关于gio/gio.h

> fatal error: gio/gio.h: No such file or directory

在正确安装opencv的依赖包和配置正确的路径之后，这个错误解决了

参考网站：

[fatal error: gio/gio.h: 没有那个文件或目录 - CSDN文库](https://wenku.csdn.net/answer/0d2d12f8f9704a56bfe4cd616b29315b)

5.关于TIFF

> Could NOT find TIFF (missing: TIFF_LIBRARY TIFF_INCLUDE_DIR)

配置正确的环境路径后这个错误解决了

参考网站：

[Ubuntu / Windows下安装Libtiff库_tiff库下载-CSDN博客](https://blog.csdn.net/qq_30354455/article/details/90757239)

6.关于编译opencv时的奇怪错误

> error: HAVE_INTROSPECTION does not appear in AM_CONDITIONAL

Opencv的依赖软件安装全了这个错误就解决了

参考网站：

["error: HAVE_INTROSPECTION does not appear in AM_CONDITIONAL" when compiling in Ubuntu · Issue #31 · solus-project/budgie-desktop (github.com)](https://github.com/solus-project/budgie-desktop/issues/31)

7.关于libavresample的报错

> -- Checking for module 'libavresample' 
>
> -- No package 'libavresample' found

在编译安装`ffmpeg`中添加

> ./configure --enable-libavresample

但是这种添加会报错，因为在新的ffmpeg中libavresample不再支持，被其他软件取代，因此即使不进行指定，ffmpeg也检测不到。

在Ubuntu22.04中无法安装libavresample，虽然Ubuntu20.04仍然可以安装这个软件，只是ffmpeg无法检测。

8.关于libdc1394-2之类的包

> No package 'libdc1394-2' found

这个包在Ubuntu22.04中无法安装，但是在ubuntu20.04可以进行安装，可以像之前在sources.txt文件中添加源，就可以下载。

参考网站：

[opencv编译问题处理集_no package 'libdc1394-2' found-CSDN博客](https://blog.csdn.net/weixin_34910922/article/details/118095033)
