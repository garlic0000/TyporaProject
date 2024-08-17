# AU-aware graph convolutional network for Macro- and Micro-expression spotting

基于AU感知的宏表情和微表情识别的图卷积网络  

DOI:10.48550/arXiv.2303.09114  

## 论文详细分析

### 摘要

> 建模面部感兴趣区域（ROI）之间的关系来提取更精细的空间特征

1.如何建模建立关系？  

2.什么是空间特征？  

参考:

[GCN—图卷积神经网络理解_gcn提取空间特征-CSDN博客](https://blog.csdn.net/wsp_1138886114/article/details/100709692)

[机器学习术语表 | Freeopen](https://freeopen.github.io/glossary/)

这里的空间特征是机器学习中的概念。

对于图片而言，中心点像素及其邻域像素构成空间，这些像素的值和关系构成特征。




> 提出了一个基于图卷积的网络，成为动作单元感知图卷积网络（AUW-GCN）

1.什么是图卷积？  

参考：

[GCN—图卷积神经网络理解_gcn提取空间特征-CSDN博客](https://blog.csdn.net/wsp_1138886114/article/details/100709692)

卷积一般指离散卷积，其本质是加权求和。

通过计算中心像素点以及相邻像素点的加权和来构成特征图谱以实现空间特征的提取。

传统的CNN（卷积神经网络）处理的图像或者视频的像素点是排列的很整齐的矩阵，但实际上有许多数据及其关系不是规整的矩阵，其网络结构为拓扑图，而GCN就是用于处理拓扑结构，用于弥补CNN无法处理拓扑图的缺陷。

图卷积是将一个节点周围的邻居按照不同的权重叠加起来，每个节点的周围的的邻居数不固定。但是一般卷积的节点的邻居数是固定的。

2.什么是基于图卷积的网络？  

参考：

[图神经网络之图卷积网络——GCN_图卷积的神经网络-CSDN博客](https://blog.csdn.net/zbp_12138/article/details/110246797)

有一些公式不太理解。

图卷积相比图像卷积更灵活。



> 为了注入先验信息和解决小数据集的问题，将AU相关的统计信息编码到网络中

1.什么是注入先验信息？先验信息是什么？

参考：

[如何将先验知识注入推荐模型-CSDN博客](https://blog.csdn.net/Kaiyuan_sjtu/article/details/121987636)

是不是预处理模型？

2.小数据集是？

参考：

[【干货指南】机器学习必须需要大量数据？小数据集也能有大价值！ (qq.com)](https://mp.weixin.qq.com/s/xGnDcRtKKt4FyVRAMPSqYA)

3.如何将AU的统计信息编码到网络中？  

参考：

[【论文阅读】AU检测|《Deep Adaptive Attention for Joint Facial Action Unit Detection and Face Alignment》_au 关键点检测-CSDN博客](https://blog.csdn.net/qq_43521527/article/details/113440970)

不太能理解。



> 在CAS(ME)2和SAMM-LV两个基准数据集上取得了新的SOTA性能

1.SOTA性能是什么?  

参考：

[论文里SOTA性能的理解-CSDN博客](https://blog.csdn.net/wuyeyoulan23/article/details/123756127)

大概是最好、最先进的意思。



### 引言

> 微表情研究的一个重要部分是微表情的定位（发现），包括在未剪辑的长视频中定位表情间隔

1.如何定位?



> 以往的研究主要使用原始特征作为输入，包括RGB图像和光流图。

1.原始特征就是指一整张图吗？



> 使用RGB样本对分别用作ME和MaE的模型输入

1.RGB样本对是？

2.MaE是宏表情吗？

3.模型输入是什么？



> RGB图像在表征面部运动，特别是微表情的面部运动方面是不够的。

1.为什么不够？





