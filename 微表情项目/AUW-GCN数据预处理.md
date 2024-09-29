# AUW-GCN数据预处理

## 实验过程

### CAS(ME)^2

进行数据预处理时，不知道使用数据集中的哪一部分，不知道是视频还是图片

#### 使用已裁剪的关键帧图片

1.`config.yaml`中的设置如下

```yaml
dataset: "cas(me)^2" 
cas(me)^2:
  dataset: "cas(me)^2"
  cropped_root_path: "/kaggle/working/cropped"
  optflow_root_path: "/kaggle/working/data/cas(me)^2/optflow_apex"
  feature_root_path: "/kaggle/working/data/cas(me)^2/feature_apex"
  feature_segment_root_path:  "/kaggle/working/data/cas(me)^2/feature_segment_apex"
  # original_anno_csv_path: "/kaggle/working/ME-GCN-Project/feature_extraction/csv/cas(me)^2.csv"
  anno_csv_path: "/kaggle/working/ME-GCN-Project/feature_extraction/csv/cas(me)^2.csv"
  CROPPED_SIZE: 500
  SEGMENT_LENGTH: 256
  RECEPTIVE_FILED: 15 # should be odd
```

2.`new_all.py`的代码如下：

```python
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

3.将`/kaggle/input/casme2/cropped/cropped`文件夹复制到`/kaggle/working`下

因为之后的操作在数据集下进行修改，在kaggle上使用`/kaggle/input`的路径不可写。

无法将修改结果记录下来。比如存放`face.csv`，`landmarks.csv`

**记录人脸和关键点**

图片数：img count =  11156

> ```bash
> /opt/conda/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
> ```

这个警告找到具体的修改位置，但是不知道是否要修改

之后修改了再看看效果

> ```bash
> /kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/SAN/lib/san_vision/transforms.py:153: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
>   img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
> ```

这个警告没有找到修改的位置，不知道影响大不大

> ```bash
> /opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:4358: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
> ```

图片完全没处理

既没有检测出人脸，也没有检测出人脸关键点

每个子文件夹都没有

甚至每张图片都没有处理

是检测模型的问题吗？

还是之前警告的问题？

> ```bash
> 0%|                                                 | 0/11156 [00:00<?, ?it/s]
> ```

代码中写了让输出检测到人脸框的图片，但是一张都没有输出，全部输出错误

> ```bash
> subject: 21, em_type: {type_item.name}, index: {index}
> ......
> ......
> subject: 19, em_type: {type_item.name}, index: {index}
> ```

**流处理**

子文件夹数：flow count =  341

对每个子文件夹进行处理（比如`anger2_5`称为一个子文件夹）

> ```bash
> denseflow "/kaggle/working/cropped/15/anger2_5" -b=10 -a=tvl1 -s=1 -if -o="/kaggle/working/data/cas(me)^2/optflow_apex/15"
> 1 videos (19 frames, 18 tvl1 flows) processed, using 0.221s, decoding speed 85.9729fps, flow speed 81.448fps
>   0%|▏                                          | 1/341 [00:01<07:24,  1.31s/it]
> 
> ```

**提取特征**

flow count =  10815

这个不知道是什么数字

全是以下错误

> ```bash
> /kaggle/working/cropped/15/disgust2_3/landmarks.csv does not exist
> /kaggle/working/cropped/15/anger2_4/landmarks.csv does not exist
> ...
> ```

就是之前人脸检测和关键点检测的部分全部失败了，所以没有任何文件

**特征分割**

由于提取特征的部分没有成功过，特征分割的部分找不到输入文件的路径。

**解决警告**

把以下这个警告进行修改，然后看看修改的效果

> ```bash
> /kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/SAN/lib/san_vision/transforms.py:153: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
>   img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
> ```

进行如下修改

```python
# 源代码
#img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes())) 
# 修改后
img = torch.ByteTensor(torch.UntypedStorage.from_buffer(pic.tobytes())) 
```

没有任何效果，人脸检测和关键点识别处还是一张图片都没法处理

不过警告消失了，不知道这个修改是否是正确的。

暂时先不改回来

#### 使用未裁剪的关键帧图片

1.`config.yaml`中的设置如下

```yaml
dataset: "cas(me)^2" 
cas(me)^2:
  dataset: "cas(me)^2"
  simpled_root_path: "/kaggle/working/selectedpic"
  cropped_root_path: "/kaggle/working/data/cas(me)^2/cropped_apex"
  optflow_root_path: "/kaggle/working/data/cas(me)^2/optflow_apex"
  feature_root_path: "/kaggle/working/data/cas(me)^2/feature_apex"
  feature_segment_root_path:  "/kaggle/working/data/cas(me)^2/feature_segment_apex"
  # original_anno_csv_path: "/kaggle/working/ME-GCN-Project/feature_extraction/csv/cas(me)^2.csv"
  anno_csv_path: "/kaggle/working/ME-GCN-Project/feature_extraction/csv/cas(me)^2.csv"
  CROPPED_SIZE: 500
  SEGMENT_LENGTH: 256
  RECEPTIVE_FILED: 15 # should be odd
```

2.`new_all.py`的代码如下：

```python
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

3.将`/kaggle/input/casme2/selectedpic/selectedpic`文件夹复制到`/kaggle/working`下

**裁剪**

图片数：img count =  11409

裁剪是成功的，查看输出的目录中，图片确实进行了裁剪

> ```
> retinaface: Finished loading model!
> 100%|████████████████████████████████████| 11409/11409 [00:48<00:00, 236.85it/s]
> ```

但是有以下警告

> ```bash
> /opt/conda/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
>   warnings.warn(msg)
> ```

**记录人脸和关键点**

图片数：img count =  11409

图片完全没处理

既没有检测出人脸，也没有检测出人脸关键点

每个子文件夹都没有

甚至每张图片都没有处理

是检测模型的问题吗？

> ```bash
> 0%|                                                 | 0/11156 [00:00<?, ?it/s]
> ```

代码中写了让输出检测到人脸框的图片，但是一张都没有输出，全部输出错误

> ```bash
> subject: 21, em_type: {type_item.name}, index: {index}
> ......
> ......
> subject: 19, em_type: {type_item.name}, index: {index}
> ```

**流处理**

子文件夹数：flow count =  357

总共有357段视频

对每个子文件夹进行处理（比如`disgust1_1`称为一个子文件夹）

> ```bash
> denseflow "/kaggle/working/data/cas(me)^2/cropped_apex/s26/disgust1_1" -b=10 -a=tvl1 -s=1 -if -o="/kaggle/working/data/cas(me)^2/optflow_apex/s26"
> 1 videos (27 frames, 26 tvl1 flows) processed, using 0.372s, decoding speed 72.5806fps, flow speed 69.8925fps
>   0%|                                           | 1/357 [00:00<04:10,  1.42it/s]
> 
> ```

**提取特征**

flow count =  11052

这个不知道是什么数字

进行裁剪的图片有11409张，11409-11052=357，总共有357段视频

全是以下错误

> ```bash
> /kaggle/working/data/cas(me)^2/cropped_apex/s26/happy2_2/landmarks.csv does not exist
> /kaggle/working/data/cas(me)^2/cropped_apex/s26/disgust2_2/landmarks.csv does not exist
> ...
> ```

就是之前人脸检测和关键点检测的部分全部失败了，所以没有任何文件

在裁剪的同时记录下面部的位置和面部的关键点的位置，在进行特征提取时需要用到

所以不管怎样，要进行特征提取，还是要有面部位置和面部关键点的位置，否则进行特征提取时会报错

所以还是检测模型的问题，没有检测到人脸和关键点

**特征分割**

由于提取特征的部分没有成功过，特征分割的部分找不到输入文件的路径。

#### 使用retina-face和dlib算法

**修改人脸检测和关键点检测函数**

使用python库retina-face进行人脸框的检测

使用dlib进行人脸关键点检测

但是dlib也可以进行人脸框的检测

到时候再试一试

而且在进行裁剪时也用到人脸框的检测，但是人脸框检测不会把人脸全部放在框内，导致裁剪后的人脸不全

以至于后续的人脸检测和关键点检测出现问题

retinaface是否能在检测人脸时把全部人脸都包括在内，特别是人脸关键点

dlib在检测人脸时是否也能包括在内，在看效果图时倒是没有在内，但是关键点检测都完全检测得到，不过进行了向上插值，要是效果明显，也去试试

人脸检测修改为：https://github.com/serengil/retinaface

```python
class FaceDetector:
    def __init__(self, img):
        """
        img 或者img_path
        """
        self.det = RetinaFace.detect_faces(img)

    def cal(self):
        """
        数据集中 图片中只有一张脸
        """
        left, top, right, bottom = self.det.get("face_1").get("facial_area")
        return left, top, right, bottom
```

还需要进行调整，比如导入模型什么的

关键点检测修改为：https://juejin.cn/post/7087767376241885215

dlib的安装居然还有问题

在python3.10的环境上还要编译，直接下载whl文件：https://pypi.org/project/dlib-bin/#files

使用dlib的精简版，即预编译版本，dlib-bin。支持人脸检测和关键点检测

```python
class LandmarkDetector:
    def __init__(self, predictor_path):
        # 检测人脸框
        self.detector = dlib.get_frontal_face_detector()
        # 下载人脸关键点检测模型： http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        # 解压为 shape_predictor_68_face_landmarks.dat
        # predictor_path = './model/shape_predictor_68_face_landmarks.dat'
        # 检测人脸关键点
        self.predictor = dlib.shape_predictor(predictor_path)



    def cal(self, img):
        # 可能已经使用cv2读好了
        # img = cv2.imread(img_path)
        # # 1 表示图像向上采样一次，图像将被放大一倍，这样可以检测更多的人脸
        for k, d in enumerate(self.detector(img)):
            # Get the landmarks/parts for the face in box d.
            shape = self.predictor(img, d)
            x_list = []
            y_list = []
            for p in shape.parts():
                x_list.append(p.x)
                y_list.append(p.y)
            # print(dir(shape))  # 'num_parts', 'part', 'parts', 'rect'
            # print(shape.num_parts)  # 68   打印出关键点的个数
            # print(shape.rect)  # 检测到每个面部的矩形框 [(118, 139) (304, 325)]
            # print(
            #     shape.parts())  # points[(147, 182), (150, 197), (154, 211), (160, 225),...,(222, 227), (215, 228)]   # 68个关键点坐标
            # # print(type(shape.part(0)))  # <class 'dlib.point'>
            # # 打印出第一个关键点和第2个关键点的坐标
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
```

在运行前安装了以下依赖

```bash
!pip install --upgrade pip
!pip install wurlitzer
!pip install retina-face
# pip install dlib 花费时间太长
# 使用精简版
!pip install dlib-bin
```

由于retina依赖的tensorflow的版本较低，导致系统的tensorflow和cuda的版本好像不匹配

> ```bash
> 2024-09-18 02:08:23.851130: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
> 2024-09-18 02:08:23.851284: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
> 2024-09-18 02:08:24.008172: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
> ```

这个真不知道咋解决，难道统一使用dlib库？

而且代码需要改一改

人脸检测的代码需要改一改，先统一导入模型，在进行下一步操作

```python
class FaceDetector:
    def __init__(self):
        """
        img 或者img_path
        """
        # 加载模型
        self.det = RetinaFace.build_model()

    def cal(self, img):
        """
        数据集中 图片中只有一张脸
        """
        faces = self.det.detect_faces(img)
        left, top, right, bottom =faces.get("face_1").get("facial_area")
        return left, top, right, bottom
```

其他的输出没有太大区别，还是裁剪后的图片中人脸检测和关键点检测不到

裁剪时，不是一张一张的裁剪的？

如果是，那怎么没报错？

可以使用之前的人脸剪切算法来试一下

> ```bash
> 0%|▏                                     | 41/11409 [00:31<1:42:16,  1.85it/s]打印检测到的人脸
> {'face_1': {'score': 0.9995631575584412, 'facial_area': [185, 127, 394, 400], 'landmarks': {'right_eye': [244.33081, 236.05351], 'left_eye': [342.47427, 245.25523], 'nose': [287.75058, 299.19818], 'mouth_right': [247.72063, 340.5572], 'mouth_left': [316.82877, 348.0262]}}}
> 1%|▏                                       | 63/11409 [00:31<58:08,  3.25it/s]打印检测到的人脸
> {'face_1': {'score': 0.9995520710945129, 'facial_area': [188, 121, 398, 391], 'landmarks': {'right_eye': [245.4684, 229.65936], 'left_eye': [344.48422, 239.0922], 'nose': [289.21347, 288.62375], 'mouth_right': [251.47635, 331.76907], 'mouth_left': [321.56885, 339.48175]}}}
> 1%|▌                                      | 152/11409 [00:31<14:18, 13.11it/s]打印检测到的人脸
> {'face_1': {'score': 0.9995953440666199, 'facial_area': [186, 129, 393, 400], 'landmarks': {'right_eye': [243.61938, 235.05725], 'left_eye': [340.3879, 246.27716], 'nose': [284.24164, 297.45468], 'mouth_right': [245.34544, 338.85837], 'mouth_left': [313.74646, 348.10666]}}}
> ```

因为第一张进行尺寸的确定

后续几张按第一张的尺寸进行裁剪

裁剪之后的图片检测不到人脸和关键点

> ```
> 打印检测到的人脸
> {}
> subject: s21, em_type: {type_item.name}, index: {index}
> 打印检测到的人脸
> {}
> subject: s21, em_type: {type_item.name}, index: {index}
> ```

还有唯一一个检测到的

> ```bash
> 打印检测到的人脸
> {'face_1': {'score': 0.9214534163475037, 'facial_area': [10, 0, 152, 193], 'landmarks': {'right_eye': [43.58823, 86.241585], 'left_eye': [99.49128, 86.49954], 'nose': [64.46772, 131.62907], 'mouth_right': [48.63125, 151.72911], 'mouth_left': [95.02033, 151.57889]}}}
> 10 0 152 193
> ```

之后就是一模一样的错误了

**错误修正**

原来是人脸框的设置出错

我以为那四个数字就是左、上、右、下

修改之后的代码

```bash
class FaceDetector:
    def __init__(self):
        """
        img 或者img_path
        """
        # 加载模型
        self.model = RetinaFace.build_model()

    def cal(self, img):
        """
        数据集中 图片中只有一张脸
        """
        faces = RetinaFace.detect_faces(img)
        print("打印检测到的人脸")
        print(faces)
        # x,y 左上坐标
        # w,h 人脸的宽高
        x, y, w, h =faces.get("face_1").get("facial_area")
        left, top, right, bottom = x, y, x+w, y+h
        return left, top, right, bottom
```

修正之后，剪切的图片可以检测到人脸了

> ```
> 打印检测到的人脸
> {'face_1': {'score': 0.9948156476020813, 'facial_area': [10, 16, 225, 299], 'landmarks': {'right_eye': [65.150635, 115.44769], 'left_eye': [168.91722, 113.176926], 'nose': [118.37486, 178.87247], 'mouth_right': [79.41906, 229.91177], 'mouth_left': [160.64532, 227.53674]}}}
> ```

但是关键点的检测有问题，检测到的面部矩形框为负数？

> ```
> 打印面部矩形框
> [(-36, 50) (251, 308)]
> 打印关键点个数
> 68
> ```

可能之前剪切的图片在剪之前，左右两边需要扩充裁剪区域

或者换SAN关键点检测算法

**左右两侧进行扩充**

```python
# padding = 100
clip_left = face_left -100
clip_right = face_right + 100
clip_top = face_top
clip_bottom = face_bottom
```

进行这样的更改之后

有两个变化

一是人脸裁剪时输出的间隔变少了？二是关键点检测打印的矩形框变为正数

但是还是有问题

这下来打印关键点的坐标

打印关键点的坐标没问题，是原函数没返回

接下来再进行测试

取消两侧扩充，进行测试，但是取消扩充会输出负数

那么两侧肯定要扩充，但是具体扩充多少还需测试

左右分别加100，右侧特别多

进行如下修改

```
padding_left = 80
padding_right = 50
```

出现了如下提醒，运行到这停止运行了

> ```bash
> ================ feature ================
> dataset: cas(me)^2
> flow count =  11052
> opt_step: 1
>  61%|███████████████████████▎              | 6774/11052 [05:02<02:49, 25.29it/s]/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy3_1/landmarks.csv does not exist
>  63%|███████████████████████▉              | 6952/11052 [05:11<03:05, 22.15it/s]/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy1_1/landmarks.csv does not exist
>  72%|███████████████████████████▍          | 7970/11052 [05:57<02:16, 22.55it/s]
> ```

确实csv文件难道是之前有的图片的关键点未检测出？

右边还是加多了，进行如下修改

```
padding_left = 80
padding_right = 10
```

有一个奇怪的现象，在进行剪切时，进度条不是根据图片的张数一张一张的进行更新

但是在进行人脸记录和关键点检测时，进度条是根据图片的张数一张一张的更新

当需要打印的字符变少时，进度条是一张一张更新，但是打印字符与进度条的更新不同步

在进行特征提取时，有几个文件夹的图片的特征没法提取，可能需要另外处理，总共有24个

```bash
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_9/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_8/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s24/happy4_4/landmarks.csv does not exist
```

但是不影响最终结果。

最终还是输出了预处理的npz文件，但是命名有问题，而且测试集为空

裁剪部分，左边往左填充80后再裁剪，右边不需要填充，甚至可能还需要去掉部分，再进行裁剪

右侧进行20的裁剪，进行如下修改

```python
padding_left = 80
# padding_right = 10
cutting_right = 20
clip_left = face_left - padding_left
clip_right = face_right - cutting_right
```

部分图片没法进行人脸记录和关键点记录，让其输出具体的数据，看看到底是人脸无法记录还是关键点无法检测

命令问题得从标注文件相关的代码进行修改

关于测试集，源项目中的测试集似乎和训练集一样，都是命名相同的.npz文件

但是源代码中，测试集保存为npy文件

**测试人脸记录和关键点检测**

现在有一个想法，就是把出问题的图片所在的目录下的所有图片都不进行裁剪，直接放入已裁剪的文件夹中，直接进行人脸检测和关键点记录，不知是否可行。

可能对每个出问题的图片所在的文件夹分别处理

裁剪的尺寸分别不一样，因为最初的图片都能检测出人脸，裁剪后检测不出人脸，但是关键点还是能检测出来。

设置右侧裁剪40

```python
padding_left = 80
# padding_right = 10
cutting_right = 40
clip_left = face_left - padding_left
clip_right = face_right - cutting_right
```

但是进行这样的处理之后

没有检测出人脸的文件夹有24个，与之前的相同，可能头的上部和下部也需要进行处理

```bash
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_8/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_9/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s24/happy4_4/landmarks.csv does not exist
```

首先应该不处理，记下有多少图片在裁剪后人脸无法检测，然后在进行调整

```python
def solve_img_size(subitem, typeitem):
    """
    处理不同图片的尺寸
    padding_top 向上填充 -
    padding_bottom 向下填充 +
    padding_left 向左填充 -
    padding_right 向右填充 +
    """
    # 首先应测试 不进行任何填充 图片有多少能检测成功
    padding_top, paddding_bottom, padding_left, padding_right = 0, 0, 0, 0
    # 左-80 右-40的情况下
    # s27 21张图片有问题
    # s37  2张图片有问题
    # s24  1张图片有问题
    # if subitem == "s":
    #     if typeitem == "happy":
    #         right_cutting = 0
    #         return right_cutting
    # else:
    #     return 40
    return padding_top, paddding_bottom, padding_left, padding_right
```

不进行处理时，再次进行关键点检测时会出现负数。人脸检测也没法成功检测

而且在特征提取部分会报错

> ```bash
> Traceback (most recent call last):
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_all.py", line 37, in <module>
>     feature(opt)
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_feature.py", line 90, in feature
>     ior_feature_list = calculate_roi_freature_list(
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/tools.py", line 354, in calculate_roi_freature_list
>     ior_flows = get_rois(
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/tools.py", line 240, in get_rois
>     return np.stack(roi_list, axis=0)
>   File "/opt/conda/lib/python3.10/site-packages/numpy/core/shape_base.py", line 449, in stack
>     raise ValueError('all input arrays must have the same shape')
> ValueError: all input arrays must have the same shape
> ```

首先进行左移80，即

```python
def solve_img_size(subitem, typeitem):
    """
    处理不同图片的尺寸
    padding_top 向上填充 -
    padding_bottom 向下填充 +
    padding_left 向左填充 -
    padding_right 向右填充 +
    """
    # 首先应测试 不进行任何填充 图片有多少能检测成功
    padding_top, padding_bottom, padding_left, padding_right = 0, 0, 0, 0
    padding_left = 80
    # 左-80 右-40的情况下
    # s27 21张图片有问题
    # s37  2张图片有问题
    # s24  1张图片有问题
    # if subitem == "s":
    #     if typeitem == "happy":
    #         right_cutting = 0
    #         return right_cutting
    # else:
    #     return 40
    return padding_top, padding_bottom, padding_left, padding_right
```

没法检测人脸的有以下文件夹

```
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_8/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_9/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s24/happy4_4/landmarks.csv does not exist
```

可能是脸的上部裁剪过多，所以检测不到脸

> ```
> # 左-80 右-40的情况下
> # s27 21张图片有问题
> # s37  2张图片有问题
> # s24  1张图片有问题
> ```

> ```
> # 左-80 的情况下
> # s27 21张图片有问题
> # s37  2张图片有问题
> # s24  1张图片有问题
> ```

对于`s24/happy4_4/img_00009.jpg`而言，脸部完全在图片内，发际线在图片上边缘，下巴的界限有点难辨认

对`s24/happy4_4`，进行上部填充10，下部填充10，其余保持不变

只进行裁剪、人脸检测和关键点检测，不知道会不会对光流提取产生影响

修改代码如下

```python
def solve_img_size(subitem, typeitem):
    """
    处理不同图片的尺寸
    padding_top 向上填充 -
    padding_bottom 向下填充 +
    padding_left 向左填充 -
    padding_right 向右填充 +
    """
    # 首先应测试 不进行任何填充 图片有多少能检测成功
    padding_top, padding_bottom, padding_left, padding_right = 0, 0, 0, 0
    padding_left = 80
    # s24/happy4_4/img_00009.jpg 脸部裁剪后 无法检测人脸
    if subitem == "s24" and typeitem == "happy4_4":
        padding_top = 10
        padding_bottom = 10
    return padding_top, padding_bottom, padding_left, padding_right
```

```bash
# apex_sampling(opt)
print("================ crop ================")
crop(opt)
print("================ record ================")
record_face_and_landmarks(opt)
# print("================ optical flow ================")
# optflow(opt)
# print("================ feature ================")
# feature(opt)
# print("================ feature segment ================")
# segment_for_train(opt)
# segment_for_test(opt)
```

还是不行，那么上下分别增加至20

```python
if subitem == "s24" and typeitem == "happy4_4":
    padding_top = 20
    padding_bottom = 20
    print(padding_top, padding_bottom)
```

需要打印出相关信息，来看看是否修改成功

之前的修改未成功，因为代码写错了，进行以下修改后，改为20的可以检测了，再进行测试增加10的，可能一开始就成功了

```python
if subitem.name == "s24" and typeitem.name == "happy4_4":
    padding_top = 10
    padding_bottom = 10
    print("测试测试测试测试测试测试测试")
    print(subitem.name, typeitem.name)
    print(padding_top, padding_bottom)
```

上下增加10可以检测人脸，但是如果需要更细节的尺寸还需要进一步测试

接下来解决两个`s37`的图片

对`s37/happy1_1/img_00003.jpg`而言，有一部分额头没有出现，需要往上填补

`s37/happy3_1/img_00001.jpg`而言，也是有一部分额头没有出现，同样需要填补

修改代码如下

```python
elif subitem.name == "s37" and (typeitem.name == "happy1_1" or typeitem.name == "happy3_1"):
    padding_top = 10
```

`happy1_1`完成了，但是`happy3_1`有新的图片的额头需要填补，修改代码如下

```python
elif subitem.name == "s37" and typeitem.name == "happy1_1":
    padding_top = 10
elif subitem.name == "s37" and typeitem.name == "happy3_1":
    padding_top = 20
```

每次只修改一个文件夹太麻烦。

同时解决`s27`的问题

对于`s27/happy3_3/img_00001.jpg`而言

额头和下巴都有点问题，因此，上下都加上10，虽然这里有21个文件夹的图片有问题，但是还是将整个s27都加上

处理的代码如下

```python
elif subitem.name == "s27":
    padding_top = 10
    padding_bottom =10
```

但是

```bash
/kaggle/working/selectedpic/s27/disgust2_7
    crop(opt)
  File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_crop.py", line 132, in crop
    cv2.imwrite(os.path.join(
cv2.error: OpenCV(4.10.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:798: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'
```

进行如下处理

```python
# s27 原图 头部太靠上
    s27_dir_list = ['happy3_3', 'happy1_6', 'happy1_5', 'happy3_2', 'happy1_1', 'anger1_3', 'disgust2_1', 'disgust2_8', 'happy2_1', 'disgust2_6', 'disgust2_9', 'disgust2_7', 'disgust2_4', 'disgust2_5', 'happy2_3', 'happy3_1', 'happy1_2', 'happy2_2', 'happy1_4', 'happy1_7', 'anger1_1']
    if subitem.name == "s24" and typeitem.name == "happy4_4":
        padding_top = 10
        padding_bottom = 10
    elif subitem.name == "s37" and typeitem.name == "happy1_1":
        padding_top = 10
    elif subitem.name == "s37" and typeitem.name == "happy3_1":
        padding_top = 20
    elif subitem.name == "s27" and typeitem.name in s27_dir_list:
        # 对于s27而言 未剪切的图片中, 头发部分几乎没出现
        # 这里的处理还得
        padding_top = -1 # 一个标志
        padding_bottom = 10
```

```python
# 对s27的处理
    if padding_top == -1:
    clip_top = 0
```

全部可以识别

接下来将右侧进行裁剪，右侧裁剪40，代码如下

```
# 右侧的要往左移 40 因此是 -40
padding_right = -40
```

这样更改后，全部都可以识别，接下来测试下光流检测是否可以全部识别

关于测试集，要不就复制训练集的？感觉没这么好处理，还要设置训练集的地址？

那还是先测试吧

```python
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

测试完成，但是文件的分割有问题

关于命名，应该在最后进行特征分割的时候，在保存时进行重命名，需要标注文件xlsx和csv的转换

在做命名转换时，我想应该把特征文件下载下来，只测试特征分割部分的代码，节省时间

接下来进行特征分割的测试，每个特征的长度为256

首先将下载的data数据传到网站，然后只保留特征分割的代码  

在上传代码时cas(me)^2中的字符不被识别，会报错，因此要改文件名

进行如下处理

```python
# apex_sampling(opt)
# print("================ crop ================")
# crop(opt)
# print("================ record ================")
# record_face_and_landmarks(opt)
# print("================ optical flow ================")
# optflow(opt)
# print("================ feature ================")
# feature(opt)
print("================ feature segment ================")
segment_for_train(opt)
segment_for_test(opt)
```

改文件名并上传文件

```yaml
# 在上传文件时 遇到了问题 只能改文件名
simpled_root_path: "/kaggle/working/selectedpic"
cropped_root_path: "/kaggle/working/data/casme_2/cropped_apex"
optflow_root_path: "/kaggle/working/data/casme_2/optflow_apex"
feature_root_path: "/kaggle/working/data/casme_2/feature_apex"
feature_segment_root_path:  "/kaggle/working/data/casme_2/feature_segment_apex"
# original_anno_csv_path: "/kaggle/working/ME-GCN-Project/feature_extraction/csv/cas(me)^2.csv"
anno_csv_path: "/kaggle/working/ME-GCN-Project/feature_extraction/csv/cas(me)^2.csv"
```

进行输出测试

```python
feature_name = os.path.split(feature_path)[-1]
video_name = os.path.splitext(feature_name)[0]
print("feature_name")
print(feature_name)
print("video_name")
print(video_name)
# 这是
tmp_df = anno_df[anno_df['video_name'] == video_name]
print("tmp_tf")
```

> ```bash
> feature_name
> anger1_1.npy
> video_name
> anger1_1
> tmp_tf
> Empty DataFrame
> Columns: [subject, video_name, start_frame, apex_frame, end_frame, type_idx, au]
> Index: []
> ```

可能改名要从一开始改，将`anger1_1`、`anger1_2`之类的图片全部归类为`anger1`并改名为`0401`

将`anger2_1``anger2_2`之类目录下的图片全部归类为`anger2`并改名为`0402`

而且我觉得可能不知要抽取关键帧，而是要抽取全部的帧，因为在特征分段时根据帧的数量分段

不过也有可能不是，所以先改名吧

改名代码如下

```python
import os
import glob
import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def changeFilesWithCSV(opt):
    try:
        simpled_root_path = opt["simpled_root_path"]
        dataset = opt["dataset"]
    except KeyError:
        print(f"Dataset {dataset} does not need to be cropped")
        print("terminate")
        exit(1)
    ch_file_name_dict = {"disgust1": "0101", "disgust2": "0102", "anger1": "0401", "anger2": "0402",
                         "happy1": "0502", "happy2": "0503", "happy3": "0505", "happy4": "0507", "happy5": "0508"}
    for sub_item in Path(simpled_root_path).iterdir():
        # sub_item s14
        if not sub_item.is_dir():
            continue
        # type_item anger1_1
        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue
            # 获取当前
            for filename in ch_file_name_dict.keys():
                # anger1 anger1_1
                if filename in type_item.name:
                    # sssss/s14/0401
                    new_dir_path = os.path.join(
                        simpled_root_path, sub_item.name, ch_file_name_dict[filename])
                    if not os.path.exists(new_dir_path):
                        os.makedirs(new_dir_path)
                    # anger1_1  0401
                    shutil.copytree(
                        str(type_item), new_dir_path, dirs_exist_ok=True)
                    # 删除 type_item 目录及其内容 递归删除
                    shutil.rmtree(type_item)
```

进行如下处理

```python
from changeFilesWithAnnoCSV import changeFilesWithCSV
# apex_sampling(opt)
# print("================ crop ================")
print("处理文件夹名称")
changeFilesWithCSV(opt)
# crop(opt)
# print("================ record ================")
# record_face_and_landmarks(opt)
# print("================ optical flow ================")
# optflow(opt)
# print("================ feature ================")
# feature(opt)
# print("================ feature segment ================")
# segment_for_train(opt)
# segment_for_test(opt)
```

再查看效果

好像可以了，但是是否要输出所有子目录来检测效果

测试好之后，还需完整走一遍流程

图片尺寸可能要重新修改

这样修改文件名称后是否会有影响？感觉光流文件变少了

当没有修改文件名称和尺寸时

> ```
> /kaggle/working/data/casme_2/cropped_apex/s27/0401/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/s27/0503/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/s27/0505/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/s27/0502/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/s27/0102/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/s37/0502/landmarks.csv does not exist
> ```

而且虽然输出，但是没有分段。

因为没有超过256帧的，但是源项目中提供的超过了256帧。

#### 使用全部未裁剪图片

那使用所有图片再操作一次。

修改如下

```python
simpled_root_path: "/kaggle/input/casme2/rawpic/rawpic"
```

所有图片有6G要是移到output不太行

可修改的路径从裁剪路径开始就可以了

但是修改文件名需要再input中操作，可能不太行

可能需要后期更改了

改名可以从裁剪时创建新的路径时修改

但是特征分割时，还有些参数不清楚

改名的代码 如下

```python
# 在这里修改
# s15 15_0101
# casme_015,casme_015_0401
# subject video_name
# 将type_item改为别的
# s15 casme_015
# /kaggle/input/casme2/rawpic/rawpic/s15/15_0101disgustingteeth
s_name = "casme_0{}".format(sub_item.name[1:-1])
v_name = "casme_0{}".format(type_item.name[0:7])
new_dir_path = os.path.join(
cropped_root_path, s_name, v_name)
```

之后可能会有图片的尺寸的问题

只进行图片裁剪，要10个小时。所以每个功能分别进行。

> ```bash
> /kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0507
> 36927.1s	436	
> 36927.1s	437	
> 36927.1s	438	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0507
> 36927.1s	439	
> 36927.1s	440	
> 36927.1s	441	该路径的图片裁剪和关键点检测出错
> 36927.1s	442	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0402/img_00001.jpg
> 36927.1s	443	检测到的脸部区域:
> 36927.1s	444	{}
> 36927.1s	445	面部矩形框
> 36927.1s	446	[(79, 50) (337, 308)]
> 36927.1s	447	
> 36927.1s	448	
> 36927.1s	449	该路径的图片裁剪和关键点检测出错
> 36927.1s	450	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0508/img_00001.jpg
> 36927.1s	451	检测到的脸部区域:
> 36927.1s	452	{}
> 36927.1s	453	面部矩形框
> 36927.1s	454	[(79, 50) (337, 308)]
> 36927.1s	455	
> 36927.1s	456	
> 36927.1s	457	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0101
> 36927.1s	458	
> 36927.1s	459	
> 36927.1s	460	该路径的图片裁剪和关键点检测出错
> 36927.1s	461	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0505/img_00001.jpg
> 36927.1s	462	检测到的脸部区域:
> 36927.1s	463	{}
> 36927.1s	464	面部矩形框
> 36927.1s	465	[(79, 50) (337, 308)]
> 36927.1s	466	
> 36927.1s	467	
> 36927.1s	468	该路径的图片裁剪和关键点检测出错
> 36927.1s	469	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0502/img_00001.jpg
> 36927.1s	470	检测到的脸部区域:
> 36927.1s	471	{}
> 36927.1s	472	面部矩形框
> 36927.1s	473	[(79, 50) (337, 308)]
> 36927.1s	474	
> 36927.1s	475	
> 36927.1s	476	/kaggle/working/data/casme_2/cropped_apex/casme_033/casme_033_0402
> 36927.1s	477	
> 36927.1s	478	
> 36927.1s	479	/kaggle/working/data/casme_2/cropped_apex/casme_033/casme_033_0102
> 37244.2s	480	/opt/conda/lib/python3.10/site-packages/traitlets/traitlets.py:2930: FutureWarning: --Exporter.preprocessors=["remove_papermill_header.RemovePapermillHeader"] for containers is deprecated in traitlets 5.0. You can pass `--Exporter.preprocessors item` ... multiple times to add items to a list.
> 37244.2s	481	  warn(
> 37244.2s	482	[NbConvertApp] WARNING | Config option `kernel_spec_manager_class` not recognized by `NbConvertApp`.
> 37244.2s	483	[NbConvertApp] Converting notebook __notebook__.ipynb to notebook
> ```

运行完成，要找出人脸检测和关键点识别有问题的

但是还是先走一遍流程，看看大概要花多长时间

大概需要16小时，有一些图片的裁剪有问题，不知道加上这些图片之后时间是否会延长

s27下的每一个子文件的图片裁剪有问题

s37下有四个文件有问题

由于每个视频中的帧都是和第一帧的截取位置相同，所以主要是第一帧

s27未裁剪图片中面部额头部分有一些不全，所以上部不进行裁剪，使用原来的高度，下巴部分不知道是否需要填充，但是还是先填充10，进行测试后再调整

s37中整个人脸都在图片中，所以需要往上移一些，按照之前的填充尺寸，选择在上部填充20，s37有6个文件夹，其中4个有问题

s21有两个文件夹，一个有问题，s21和s27的问题一样，未裁剪图片中本身额头部分不全，因此上部不裁剪，使用原来的高度，但是下巴部分的空间很充分，不需要进行填充

s31有7个文件夹，有一个文件夹有问题，但是未裁剪的图片中整张脸都显示出来了，所以需要在上部填充。先填充10进行测试

> ```
> /kaggle/working/data/casme_2/cropped_apex/casme_021/casme_021_0401/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0401/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0503/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0102/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0402/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0507/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0502/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0505/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0101/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0508/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0502/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0505/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0402/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0508/landmarks.csv does not exist
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0507/landmarks.csv does not exist
> ```

调整的代码如下

> ```python
> # s27 原图 头部太靠上
>     if subitem.name == "s31" and typeitem.name == "31_0507climbingthewall":
>         padding_top = 10
>         padding_bottom = 10
>     elif subitem.name == "s37":
>         padding_top = 20
>     # ch_file_name_dict = {"disgust1": "0101", "disgust2": "0102", "anger1": "0401", "anger2": "0402",
>     #                          "happy1": "0502", "happy2": "0503", "happy3": "0505", "happy4": "0507", "happy5": "0508"}
>     # "happy1": "0502", "happy2": "0503", "happy3": "0505"
>     # "anger1": "0401"
>     # "disgust2": "0102"
>     elif subitem.name == "s27":
>         # 对于s27而言 未剪切的图片中, 头发部分几乎没出现
>         # 这里的处理还得
>         padding_top = -1 # 一个标志
>         padding_bottom = 10
>     elif subitem.name == "s21":
>         # 对于s27而言 未剪切的图片中, 头发部分几乎没出现
>         # 这里的处理还得
>         padding_top = -1 # 一个标志
> ```

可能要专门写一个检测图片是否能正常检测的程序

```python
def check_crop(img, img_path):
    """
    检测裁剪后的图片是否能再次检测人脸
    用于调整裁剪尺寸用
    """
    face_detector = FaceDetector()
    try:
        face_left, face_top, face_right, face_bottom = \
            face_detector.cal(img)
    except Exception:
        print("\n")
        print("该路径的图片裁剪出错")
        print(img_path)
        face_detector.info(img)
```

```python
# 用于调错
# 检测裁剪后的图片是否能检测到人脸
check_crop(img, img_path)
# 不写 只测试
# cv2.imwrite(os.path.join(
#             new_dir_path,
#             f"img_{str(index+1).zfill(5)}.jpg"), img)
```

但是这样会运行多久其实也不清楚，之前人脸检测和关键点检测有几乎8个小时

进行检测用于调错也需要8小时，所以不进行，直接人脸检测和关键点检测

将调错函数注释，裁剪图片的写入函数取消注释

运行有有一个检测出错

> ```bash
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0508/img_00111.jpg
> ```

这个照片在裁剪后，下巴不全，所以应该，往下再填补一些，代码修改如下

```python
 elif subitem.name == "s27":
        # 对于s27而言 未剪切的图片中, 头发部分几乎没出现
        # 这里的处理还得
        padding_top = -1 # 一个标志
        padding_bottom = 20
```

要不只对这一个视频进行测试，就是casme_027_0508，这样就不会出那样的错。花费的时间也比较少。

测试，代码如下

在输出目录创建如下代码

```bash
# 用于调试
!mkdir -p /kaggle/working/rawpic/s27
!cp /kaggle/input/casme2/rawpic/rawpic/s27/27_0508funnydunkey /kaggle/working/rawpic/s27/
```

更改yaml文件中的地址

```yaml
  # 用于调试错误
  # /kaggle/working/rawpic
  simpled_root_path: "/kaggle/working/rawpic"
  # simpled_root_path: "/kaggle/input/casme2/rawpic/rawpic"
```

如果测试的没问题，就可以进行晚上的所有数据集测试

进行这样的测试之后，顺便把光流测试和特征提取同时运行，确保没问题。

先将训练集的数据全部复制到测试集

复制的代码如下

```python
# 直接将训练集的数据 复制到测试集
    out_dir = os.path.join(
        feature_segment_root_path, "test")
    train_dir = os.path.join(feature_segment_root_path, "train")
    # 递归复制
    shutil.copytree(train_dir, out_dir)
    # for sub_item in Path(feature_root_path).iterdir():
    #     if not sub_item.is_dir():
    #         continue
    #     out_dir = os.path.join(
    #         feature_segment_root_path, "test", sub_item.name)
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #     for type_item in sub_item.iterdir():
    #         if not type_item.is_dir():
    #             continue
    #         video_name = type_item.name
    #         # test文件夹为空的原因是没有feature.npy文件
    #         # 测试集是直接复制？
    #         # 我觉得测试集应该从整个数据集中选取
    #         # 要保存为npz文件吗？
    #
    #         feature_path = os.path.join(str(type_item), "feature.npy")
    #         shutil.copy(feature_path, os.path.join(out_dir, video_name+".npy"))

    print("segment for test Finished!")
```

可以复制。

如果这次可以成功运行下来，直接将运行的结果用于训练

运行的环境就设置成最初的环境。看看训练的结果有没有提升。

还有可视化的结果表现。将输出的图片，所处的时间区间进行绘制。

如果运行成功，将CASME和CASMEⅡ数据集进行测试，增加论文的数据集。但是SAMM-LV的数据集不知道从哪获取。下次从老师那再催一催。

结果可以运行成功，但是效果不好。

微表情没有分数。

我的想法是要对裁剪和关键点检测的算法进行更改。

裁剪算法：

https://github.com/deepinsight/insightface

https://github.com/ipazc/mtcnn

关键点检测算法：

https://github.com/1adrianb/face-alignment

https://github.com/yfeng95/PRNet

#### 使用retinaface和san算法

**换原来的算法**

使用retinaface进行人脸检测

使用san进行人脸关键点检测

代码如下：

```python
class LandmarkDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = SanLandmarkDetector(model_path, device)

    def cal(self, img, offset=None, face_box=None):
        if face_box is None:
            face_box = (0, 0, img.shape[1], img.shape[0])
        locs, _ = self.det.detect(img, face_box)
        x_list = [
            loc[0] if offset is None else loc[0] - offset[0] for loc in locs]
        y_list = [
            loc[1] if offset is None else loc[1] - offset[1] for loc in locs]
        return x_list, y_list


class FaceDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = RetinaFaceDetector(model_path, device)

    def cal(self, img):
        left, top, right, bottom = self.det.get_face_box(img)
        return left, top, right, bottom
```

```python
face_det_model_path = "/kaggle/input/checkpoint/pytorch/default/1/retinaface_Resnet50_Final.pth"
face_detector = FaceDetector(face_det_model_path)
......
face_left, face_top, face_right, face_bottom = \
                                face_detector.cal(img)
clip_left = face_left
clip_right = face_right
clip_top = face_top
clip_bottom = face_bottom
```

```python
face_det_model_path = "/kaggle/input/checkpoint/pytorch/default/1/retinaface_Resnet50_Final.pth"
face_detector = FaceDetector(face_det_model_path)
landmark_model_path = '/kaggle/input/checkpoint/pytorch/default/1/san_checkpoint_49.pth.tar'
landmark_detector = LandmarkDetector(landmark_model_path)
......
left, top, right, bottom = face_detector.cal(img)
x_list, y_list = landmark_detector.cal(img, face_box=(left, top, right, bottom))
```

先进行人脸裁剪、人脸记录和关键点记录的代码测试

因为在裁剪之后，可能存在人脸检测失败的情况。

出现裁剪后人脸检测失败的情况如下：

> ```
> 				该路径的图片裁剪和关键点检测出错
> 2785.2s	6878	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0505/img_00001.jpg	
> 2785.2s	6881	该路径的图片裁剪和关键点检测出错
> 2785.2s	6882	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0502/img_00001.jpg
> 2785.2s	6885	该路径的图片裁剪和关键点检测出错
> 2785.2s	6886	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0401/img_00001.jpg
> 2785.2s	6889	该路径的图片裁剪和关键点检测出错
> 2785.2s	6890	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0402/img_00001.jpg
> 2785.2s	6893	该路径的图片裁剪和关键点检测出错
> 2785.2s	6894	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0507/img_00001.jpg
> 2785.2s	6897	该路径的图片裁剪和关键点检测出错
> 2785.2s	6898	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0101/img_00001.jpg
> 2785.2s	6901	该路径的图片裁剪和关键点检测出错
> 2785.2s	6902	/kaggle/working/data/casme_2/cropped_apex/casme_016/casme_016_0102/img_00001.jpg	
> 2785.2s	6905	该路径的图片裁剪和关键点检测出错
> 2785.2s	6906	/kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0503/img_00001.jpg
> 2785.2s	6909	该路径的图片裁剪和关键点检测出错
> 2785.2s	6910	/kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0102/img_00001.jpg
> 2785.2s	6913	该路径的图片裁剪和关键点检测出错
> 2785.2s	6914	/kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0401/img_00001.jpg	
> 2785.2s	6917	该路径的图片裁剪和关键点检测出错
> 2785.2s	6918	/kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0101/img_00001.jpg
> 2785.2s	6921	该路径的图片裁剪和关键点检测出错
> 2785.2s	6922	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0508/img_00001.jpg	
> 2785.2s	6925	该路径的图片裁剪和关键点检测出错
> 2785.2s	6926	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0505/img_00001.jpg
> 2785.2s	6929	该路径的图片裁剪和关键点检测出错
> 2785.2s	6930	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0502/img_00001.jpg	
> 2785.2s	6933	该路径的图片裁剪和关键点检测出错
> 2785.2s	6934	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0401/img_00001.jpg
> 2785.2s	6937	该路径的图片裁剪和关键点检测出错
> 2785.2s	6938	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0503/img_00001.jpg
> 2785.2s	6941	该路径的图片裁剪和关键点检测出错
> 2785.2s	6942	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0402/img_00001.jpg
> 2785.2s	6945	该路径的图片裁剪和关键点检测出错
> 2785.2s	6946	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0101/img_00001.jpg
> 2785.2s	6949	该路径的图片裁剪和关键点检测出错
> 2785.2s	6950	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0507/img_00001.jpg
> 2785.2s	6953	该路径的图片裁剪和关键点检测出错
> 2785.2s	6954	/kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0102/img_00001.jpg
> 2785.2s	6957	该路径的图片裁剪和关键点检测出错
> 2785.2s	6958	/kaggle/working/data/casme_2/cropped_apex/casme_038/casme_038_0502/img_00001.jpg
> 2785.2s	6961	该路径的图片裁剪和关键点检测出错
> 2785.2s	6962	/kaggle/working/data/casme_2/cropped_apex/casme_038/casme_038_0507/img_00001.jpg	
> 2785.2s	6965	该路径的图片裁剪和关键点检测出错
> 2785.2s	6966	/kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0402/img_00001.jpg
> 2785.2s	6969	该路径的图片裁剪和关键点检测出错
> 2785.2s	6970	/kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0401/img_00001.jpg
> 2785.2s	6973	该路径的图片裁剪和关键点检测出错
> 2785.2s	6974	/kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0507/img_00001.jpg
> 2785.2s	6977	该路径的图片裁剪和关键点检测出错
> 2785.2s	6978	/kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0101/img_00001.jpg
> 2785.2s	6981	该路径的图片裁剪和关键点检测出错
> 2785.2s	6982	/kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0502/img_00001.jpg
> 2785.2s	6985	该路径的图片裁剪和关键点检测出错
> 2785.2s	6986	/kaggle/working/data/casme_2/cropped_apex/casme_034/casme_034_0401/img_00001.jpg
> 2785.2s	6989	该路径的图片裁剪和关键点检测出错
> 2785.2s	6990	/kaggle/working/data/casme_2/cropped_apex/casme_034/casme_034_0503/img_00001.jpg
> 2785.2s	6993	该路径的图片裁剪和关键点检测出错
> 2785.2s	6994	/kaggle/working/data/casme_2/cropped_apex/casme_034/casme_034_0402/img_00001.jpg
> 2785.2s	6997	该路径的图片裁剪和关键点检测出错
> 2785.2s	6998	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0507/img_00001.jpg
> 2785.2s	7001	该路径的图片裁剪和关键点检测出错
> 2785.2s	7002	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0101/img_00001.jpg
> 2785.2s	7005	该路径的图片裁剪和关键点检测出错
> 2785.2s	7006	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0402/img_00001.jpg
> 2785.2s	7009	该路径的图片裁剪和关键点检测出错
> 2785.2s	7010	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0508/img_00001.jpg
> 2785.2s	7013	该路径的图片裁剪和关键点检测出错
> 2785.2s	7014	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0505/img_00001.jpg
> 2785.2s	7017	该路径的图片裁剪和关键点检测出错
> 2785.2s	7018	/kaggle/working/data/casme_2/cropped_apex/casme_037/casme_037_0502/img_00001.jpg
> 2785.2s	7021	该路径的图片裁剪和关键点检测出错
> 2785.2s	7022	/kaggle/working/data/casme_2/cropped_apex/casme_029/casme_029_0502/img_00001.jpg
> 2785.2s	7025	该路径的图片裁剪和关键点检测出错
> 2785.2s	7026	/kaggle/working/data/casme_2/cropped_apex/casme_020/casme_020_0502/img_00001.jpg
> 2785.2s	7029	该路径的图片裁剪和关键点检测出错
> 2785.2s	7030	/kaggle/working/data/casme_2/cropped_apex/casme_040/casme_040_0503/img_00001.jpg
> 2785.2s	7033	该路径的图片裁剪和关键点检测出错
> 2785.2s	7034	/kaggle/working/data/casme_2/cropped_apex/casme_040/casme_040_0401/img_00001.jpg
> 2785.2s	7037	该路径的图片裁剪和关键点检测出错
> 2785.2s	7038	/kaggle/working/data/casme_2/cropped_apex/casme_040/casme_040_0502/img_00001.jpg
> 2785.2s	7041	该路径的图片裁剪和关键点检测出错
> 2785.2s	7042	/kaggle/working/data/casme_2/cropped_apex/casme_035/casme_035_0102/img_00001.jpg
> 2785.2s	7045	该路径的图片裁剪和关键点检测出错
> 2785.2s	7046	/kaggle/working/data/casme_2/cropped_apex/casme_022/casme_022_0503/img_00001.jpg
> 2785.2s	7049	该路径的图片裁剪和关键点检测出错
> 2785.2s	7050	/kaggle/working/data/casme_2/cropped_apex/casme_022/casme_022_0402/img_00001.jpg
> 2785.2s	7053	该路径的图片裁剪和关键点检测出错
> 2785.2s	7054	/kaggle/working/data/casme_2/cropped_apex/casme_022/casme_022_0101/img_00001.jpg
> 2785.2s	7057	该路径的图片裁剪和关键点检测出错
> 2785.2s	7058	/kaggle/working/data/casme_2/cropped_apex/casme_022/casme_022_0508/img_00001.jpg
> 2785.2s	7061	该路径的图片裁剪和关键点检测出错
> 2785.2s	7062	/kaggle/working/data/casme_2/cropped_apex/casme_022/casme_022_0102/img_00001.jpg
> 2785.2s	7065	该路径的图片裁剪和关键点检测出错
> 2785.2s	7066	/kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0102/img_00001.jpg
> 2785.2s	7069	该路径的图片裁剪和关键点检测出错
> 2785.2s	7070	/kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0101/img_00001.jpg
> 2785.2s	7073	该路径的图片裁剪和关键点检测出错
> 2785.2s	7074	/kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0508/img_00001.jpg
> 2785.2s	7077	该路径的图片裁剪和关键点检测出错
> 2785.2s	7078	/kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0502/img_00001.jpg
> 2785.2s	7081	该路径的图片裁剪和关键点检测出错
> 2785.2s	7082	/kaggle/working/data/casme_2/cropped_apex/casme_023/casme_023_0503/img_00001.jpg
> 2785.2s	7085	该路径的图片裁剪和关键点检测出错
> 2785.2s	7086	/kaggle/working/data/casme_2/cropped_apex/casme_023/casme_023_0102/img_00001.jpg
> 2785.2s	7089	该路径的图片裁剪和关键点检测出错
> 2785.2s	7090	/kaggle/working/data/casme_2/cropped_apex/casme_023/casme_023_0402/img_00001.jpg
> 2785.2s	7093	该路径的图片裁剪和关键点检测出错
> 2785.2s	7094	/kaggle/working/data/casme_2/cropped_apex/casme_023/casme_023_0507/img_00001.jpg
> 2785.2s	7097	该路径的图片裁剪和关键点检测出错
> 2785.2s	7098	/kaggle/working/data/casme_2/cropped_apex/casme_036/casme_036_0401/img_00001.jpg
> 2785.2s	7101	该路径的图片裁剪和关键点检测出错
> 2785.2s	7102	/kaggle/working/data/casme_2/cropped_apex/casme_036/casme_036_0505/img_00001.jpg
> 2785.2s	7105	该路径的图片裁剪和关键点检测出错
> 2785.2s	7106	/kaggle/working/data/casme_2/cropped_apex/casme_021/casme_021_0401/img_00001.jpg
> 2785.2s	7109	该路径的图片裁剪和关键点检测出错
> 2785.2s	7110	/kaggle/working/data/casme_2/cropped_apex/casme_021/casme_021_0101/img_00001.jpg
> 2785.2s	7113	该路径的图片裁剪和关键点检测出错
> 2785.2s	7114	/kaggle/working/data/casme_2/cropped_apex/casme_033/casme_033_0102/img_00001.jpg
> 2785.2s	7117	该路径的图片裁剪和关键点检测出错
> 2785.2s	7118	/kaggle/working/data/casme_2/cropped_apex/casme_033/casme_033_0402/img_00001.jpg
> 2785.2s	7121	该路径的图片裁剪和关键点检测出错
> 2785.2s	7122	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0402/img_00001.jpg
> 2792.3s	7126	该路径的图片裁剪和关键点检测出错
> 2792.3s	7127	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0101/img_00001.jpg
> 2792.3s	7130	该路径的图片裁剪和关键点检测出错
> 2792.3s	7131	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0505/img_00001.jpg
> 2792.3s	7134	该路径的图片裁剪和关键点检测出错
> 2792.3s	7135	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0102/img_00001.jpg
> 2792.3s	7138	该路径的图片裁剪和关键点检测出错
> 2792.3s	7139	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0507/img_00001.jpg
> 2792.3s	7142	该路径的图片裁剪和关键点检测出错
> 2792.3s	7143	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0502/img_00001.jpg
> 2792.3s	7146	该路径的图片裁剪和关键点检测出错
> 2792.3s	7147	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0508/img_00001.jpg
> 2792.3s	7150	该路径的图片裁剪和关键点检测出错
> 2792.3s	7151	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0503/img_00001.jpg
> 2792.3s	7154	该路径的图片裁剪和关键点检测出错
> 2792.3s	7155	/kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0401/img_00001.jpg
> 2792.3s	7158	该路径的图片裁剪和关键点检测出错
> 2792.3s	7159	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0401/img_00001.jpg
> 2792.3s	7162	该路径的图片裁剪和关键点检测出错
> 2792.3s	7163	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0502/img_00001.jpg
> 2792.3s	7166	该路径的图片裁剪和关键点检测出错
> 2792.3s	7167	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0102/img_00001.jpg
> 2792.3s	7170	该路径的图片裁剪和关键点检测出错
> 2792.3s	7171	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0503/img_00001.jpg
> 2792.3s	7174	该路径的图片裁剪和关键点检测出错
> 2792.3s	7175	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0101/img_00001.jpg
> 2792.3s	7178	该路径的图片裁剪和关键点检测出错
> 2792.3s	7179	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0505/img_00001.jpg
> 2792.3s	7182	该路径的图片裁剪和关键点检测出错
> 2792.3s	7183	/kaggle/working/data/casme_2/cropped_apex/casme_030/casme_030_0507/img_00001.jpg
> 2792.3s	7186	该路径的图片裁剪和关键点检测出错
> 2792.3s	7187	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0505/img_00001.jpg
> 2792.3s	7190	该路径的图片裁剪和关键点检测出错
> 2792.3s	7191	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0101/img_00001.jpg
> 2792.3s	7194	该路径的图片裁剪和关键点检测出错
> 2792.3s	7195	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0402/img_00001.jpg
> 2792.3s	7198	该路径的图片裁剪和关键点检测出错
> 2792.3s	7199	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0508/img_00001.jpg
> 2792.3s	7202	该路径的图片裁剪和关键点检测出错
> 2792.3s	7203	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0401/img_00001.jpg
> 2792.3s	7206	该路径的图片裁剪和关键点检测出错
> 2792.3s	7207	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0502/img_00001.jpg
> 2792.3s	7210	该路径的图片裁剪和关键点检测出错
> 2792.3s	7211	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0102/img_00001.jpg
> 2792.3s	7214	该路径的图片裁剪和关键点检测出错
> 2792.3s	7215	/kaggle/working/data/casme_2/cropped_apex/casme_015/casme_015_0503/img_00001.jpg
> 2792.3s	7218	该路径的图片裁剪和关键点检测出错
> 2792.3s	7219	/kaggle/working/data/casme_2/cropped_apex/casme_019/casme_019_0402/img_00001.jpg
> 2792.3s	7222	该路径的图片裁剪和关键点检测出错
> 2792.3s	7223	/kaggle/working/data/casme_2/cropped_apex/casme_019/casme_019_0505/img_00001.jpg
> 2792.3s	7226	该路径的图片裁剪和关键点检测出错
> 2792.3s	7227	/kaggle/working/data/casme_2/cropped_apex/casme_019/casme_019_0102/img_00001.jpg
> 2792.3s	7230	该路径的图片裁剪和关键点检测出错
> 2792.3s	7231	/kaggle/working/data/casme_2/cropped_apex/casme_019/casme_023_0502/img_00001.jpg
> 2792.3s	7234	该路径的图片裁剪和关键点检测出错
> 2792.3s	7235	/kaggle/working/data/casme_2/cropped_apex/casme_019/casme_019_0507/img_00001.jpg
> 2792.3s	7238	该路径的图片裁剪和关键点检测出错
> 2792.3s	7239	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0101/img_00001.jpg
> 2792.3s	7242	该路径的图片裁剪和关键点检测出错
> 2792.3s	7243	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0502/img_00001.jpg
> 2792.3s	7246	该路径的图片裁剪和关键点检测出错
> 2792.3s	7247	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0507/img_00001.jpg
> 2792.3s	7250	该路径的图片裁剪和关键点检测出错
> 2792.3s	7251	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0503/img_00001.jpg
> 2792.3s	7254	该路径的图片裁剪和关键点检测出错
> 2792.3s	7255	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0505/img_00001.jpg
> 2792.3s	7258	该路径的图片裁剪和关键点检测出错
> 2792.3s	7259	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0401/img_00001.jpg
> 2792.3s	7262	该路径的图片裁剪和关键点检测出错
> 2792.3s	7263	/kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0402/img_00001.jpg
> ```

每个文件夹进行裁剪之后都没法再次进行人脸检测，可能裁剪的非常贴近人脸。需要进行进一步的调整

先把裁剪的文件夹下载下来

裁剪的图片非常贴近人脸，但是再一次检测就无法检测出来，于是进行填充

修改如下：

```python
padding_top, padding_bottom, padding_left, padding_right = \
                                solve_img_size(sub_item, type_item)
clip_top = face_top - padding_top
clip_bottom = face_bottom + padding_bottom
clip_left = face_left - padding_left
clip_right = face_right + padding_right
# 对s27 s21的处理
if padding_top == -1:
  clip_top = 0
```

```python
def solve_img_size(subitem, typeitem):
    """
    处理不同图片的尺寸
    padding_top 向上填充 -
    padding_bottom 向下填充 +
    padding_left 向左填充 -
    padding_right 向右填充 +
    """
    # 首先应测试 不进行任何填充 图片有多少能检测成功
    padding_top, padding_bottom, padding_left, padding_right = 10, 10, 10, 10
    if subitem.name == "s27" or subitem.name == "s21":
        padding_top = -1  # 一个标志
    return padding_top, padding_bottom, padding_left, padding_right
```

进行测试

还是有问题，增加填充的面积

```python
padding_top, padding_bottom, padding_left, padding_right = 20, 20, 20, 20
```

进行这样的填充后，变少了

> ```
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0101/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0503/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0402/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0507/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0401/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0505/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_031/casme_031_0502/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_040/casme_040_0502/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_040/casme_040_0503/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_040/casme_040_0401/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0503/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0507/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0102/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0101/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0402/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0401/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0508/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0502/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_027/casme_027_0505/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0101/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0502/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0402/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0401/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_024/casme_024_0507/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_033/casme_033_0402/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_033/casme_033_0102/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0507/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0101/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0401/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0502/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0402/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0102/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0505/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0508/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_032/casme_032_0503/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_035/casme_035_0102/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0102/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0508/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0502/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_025/casme_025_0101/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0401/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0102/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0101/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_026/casme_026_0503/img_00001.jpg
> 该路径的图片裁剪和关键点检测出错
> /kaggle/working/data/casme_2/cropped_apex/casme_020/casme_020_0502/img_00001.jpg
> ```

031、040、027、024、033、032、035、025、026、020需要全部重新调整

```python
    if subitem.name == "s27" or subitem.name == "s21":
        padding_top = -1  # 一个标志
    # 031、040、027、024、033、032、035、025、026、020需要全部重新调整
    elif subitem.name in ["s31", "s40", "s27", "s24", "s33", "s32", "s35", "s25", "s26", "s20"]:
        padding_top, padding_bottom, padding_left, padding_right = 30, 30, 30, 30
```

在运行前发现有问题

> ```bash
> Traceback (most recent call last):
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_crop.py", line 199, in <module>
>     crop(opt)
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_crop.py", line 181, in crop
>     cv2.imwrite(os.path.join(
> cv2.error: OpenCV(4.10.0) /io/opencv/modules/imgcodecs/src/loadsave.cpp:798: error: (-215:Assertion failed) !_img.empty() in function 'imwrite'
> ```

可能变少的原因是有一部分的人脸已经到顶了

所以接下来只进行人脸裁剪的测试

代码如下

```python
# 在进行填充时可能会超过图片尺寸
# 进行测试
if clip_top < 0 or clip_bottom < 0 or clip_left < 0 or clip_right < 0:
	print(clip_top, clip_bottom, clip_left, clip_right)
	continue
```

```python
# # 031、040、027、024、033、032、035、025、026、020需要全部重新调整
    # elif subitem.name in ["s31", "s40", "s27", "s24", "s33", "s32", "s35", "s25", "s26", "s20"]:
    #     padding_top, padding_bottom, padding_left, padding_right = 30, 30, 30, 30
```

并且将padding改为15

```python
# 首先应测试 不进行任何填充 图片有多少能检测成功
padding_top, padding_bottom, padding_left, padding_right = 15, 15, 15, 15
```

进行测试

有好几个输出是这样的

> ```
> -1 339 133 396
> -1 339 133 396
> -1 339 133 396
> -1 339 133 396
> -1 339 133 396
> -1 339 133 396
> ```

也就是说头顶的padding_top=0最好

但是这个时候，s27和s21需要修改吗？

先进行以下修改

```
padding_top, padding_bottom, padding_left, padding_right = 14, 14, 14, 14
```

进行这样的修改没有问题

测试下这样修改后，能否检测出关键点

这样还是没法检测出关键点，对两边进行扩充

两边分别扩充20

同时，检测s27和s21能否同样使用14

代码修改如下

```python
padding_top = 14
padding_bottom = 14
padding_left, padding_right = 20, 20
# if subitem.name == "s27" or subitem.name == "s21":
#     padding_top = -1  # 一个标志
```

如果不报错，则s27和s21可使用相同的处理

s27比较特殊，不能进行裁剪

> ```
> -8 349 170 444
> -1 352 168 445
> -9 351 177 454
> -5 352 169 445
> ```

还是直接使用

但是s21可以使用-14

再将两边进行扩宽，代码修改如下

```python
padding_top = 14
padding_bottom = 14
padding_left, padding_right = 30, 30
if subitem.name == "s27":
	padding_top = -1  # 一个标志
```

两边还需要进一步的扩宽

```python
padding_top = 14
padding_bottom = 14
padding_left, padding_right = 40, 40
if subitem.name == "s27":
	padding_top = -1  # 一个标志
```

修改为50

突然有一个想法，就是剪切之后没法再次检测人脸，那么该图片的长宽就是人脸的长宽。

直接使用图片的长宽

先做测试，查看人脸检测输出的格式

使用已裁剪的图像进行检测，查看输出格式

相关代码如下

```python
simpled_root_path: "/kaggle/input/casme2/cropped/cropped"
```

```python
    def cal(self, img):
        # 用于测试 输出检测格式
        self.info(img)
        left, top, right, bottom = self.det.get_face_box(img)
        return left, top, right, bottom

    def info(self, img):
        """
        用于调错
        """
        print(self.det.get_face_box(img))
```

能检测人脸，但是会有负数，比如

> ```
> (0, -3, 196, 237)
> ```

应该后两位不会是负数，所以如果前两位出现负数，则将负数改为0

修改代码如下：

```python
    def cal(self, img):
        # 用于测试 输出检测格式
        # self.info(img)
        left, top, right, bottom = self.det.get_face_box(img)
        # 检测到已裁剪的人脸图像时
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        return left, top, right, bottom
```

可以正常检测人脸了，连检测裁剪之后的人脸也可以

但是在检测关键点出现了问题

> ```bash
> Traceback (most recent call last):
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_record_face_and_landmark.py", line 99, in record_face_and_landmarks
>     x_list, y_list = landmark_detector.cal(img, face_box=(left, top, right, bottom))
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/tools.py", line 51, in cal
>     locs, _ = self.det.detect(img, face_box)
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/SAN/san_api.py", line 111, in detect
>     return locations.round().astype(np.int), scores
>   File "/opt/conda/lib/python3.10/site-packages/numpy/__init__.py", line 324, in __getattr__
>     raise AttributeError(__former_attrs__[attr])
> AttributeError: module 'numpy' has no attribute 'int'.
> `np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
> The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
>     https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'inf'?
> 
> During handling of the above exception, another exception occurred:
> ```

在`File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/SAN/san_api.py", line 111`

进行以下修改

```python
# return locations.round().astype(np.int), scores
# 随便选的精度
return locations.round().astype(np.int64), scores
```

修改后可以进行关键点检测

但是在检测关键点时，图片的数量变成

> ```bash
> img count =  3323
> ```

在检测光流时出现以下问题

> ```bash
> terminate called after throwing an instance of 'cv::Exception'
>   what():  OpenCV(4.10.0) /kaggle/working/opencv_build/opencv_contrib/modules/cudaoptflow/src/tvl1flow.cpp:188: error: (-215:Assertion failed) I0.size() == I1.size() in function 'calcImpl'
> ```

这个说明图片尺寸不一致

但是在图片裁剪时，已经一致了，为什么会出现这样的问题？

重新再运行一遍试试

可能是文件夹命名问题

先暂时进行这样的修改进行测试

```python
# s_name = "casme_0{}".format(sub_item.name[1:])
# v_name = "casme_0{}".format(type_item.name[0:7])
# new_dir_path = os.path.join(
#     cropped_root_path, s_name, v_name)
new_dir_path = os.path.join(
	cropped_root_path, sub_item.name, type_item.name)
```

进行这样的修改后文件的个数也正常了

> ```bash
> img count =  11156
> ```

也可以正常提取光流特征

将之前的测试条件修改回来，同时将修改图片尺寸的代码删除，然后进行正式的测试

```python
simpled_root_path: "/kaggle/input/casme2/rawpic/rawpic"
```

```python
s_name = "casme_0{}".format(sub_item.name[1:])
v_name = "casme_0{}".format(type_item.name[0:7])
new_dir_path = os.path.join(
	cropped_root_path, s_name, v_name)
# new_dir_path = os.path.join(
#     cropped_root_path, sub_item.name, type_item.name)
```

运行成功了，而且比之前的速度快，存储占用的少，所以可能在预处理这一块没法优化了

因为输出的内存少，所以应该可以一次性运行完

在提取特征时需要了问题，出现了下面的错误

> ```bash
> Traceback (most recent call last):
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_feature.py", line 139, in <module>
>     feature(opt)
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_feature.py", line 90, in feature
>     ior_feature_list = calculate_roi_freature_list(
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/tools.py", line 296, in calculate_roi_freature_list
>     global_optflow_vector = cal_global_optflow_vector(flow, landmarks)
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/tools.py", line 266, in cal_global_optflow_vector
>     flow_nose_roi = get_main_direction_flow(
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/tools.py", line 207, in get_main_direction_flow
>     for i, ang in enumerate(angs):
> TypeError: 'NoneType' object is not iterable
> ```

提取光流和关键点检测都没报错，但是不排除是否是关键点检测的为空，或者有负数

首先修改下图片中人脸检测的代码

有时人脸检测框会出现上和左为负数，但是又没有可能会出现右和下大于图片的宽度和高度，但是在裁剪时没有报错，那应该是没有出现，但是还是进行修改。

即出现上或左为负数，或者，右大于宽度或下大于高度，则使用源图像的尺寸，不进行裁剪，或者不进行新的人脸记录

代码如下

```python
# 检测到已裁剪的人脸图像 检测的参数不合法时
        if left < 0 or top < 0 or right > img.shape[1] or bottom > img.shape[0]:
            left, top, right, bottom = 0, 0, img.shape[1], img.shape[0]
```

使用出问题的进行测试

在进行特征分段时，有一个文件夹没有特征，可能是在提取特征时出现了问题，只创建了文件夹没有提取特征，所以在进行特征分段时为空文件夹

> ```bash
> feature_name
> casme_023_0503.npy
> video_name
> casme_023_0503
> tmp_tf
> Empty DataFrame
> Columns: [subject, video_name, start_frame, apex_frame, end_frame, type_idx, au]
> Index: []
> frame_count
> 2813
> ```

原始文件夹有2814张图片但是这里只有2813张，可能有一张关键点检测失败导致特征提取失败

把这个有问题的文件夹移到/kaggle/working/下，只对其进行测试，主要测关键点的输出格式

代码如下

```bash
# 用于调试
!mkdir -p /kaggle/working/rawpic/s23
!cp -r /kaggle/input/casme2/rawpic/rawpic/s23/23_0503unnyfarting /kaggle/working/rawpic/s23/
```

```python
locs, _ = self.det.detect(img, face_box)
# 用于测试
print(locs)
```

```python
simpled_root_path: "/kaggle/working/rawpic"
```

在训练时总是结束不了

改为以下代码，让输出简洁一些

```python
# 用于测试
print(len(locs))
```

在测试时输出

> ```bash
> feature_name
> casme_023_0503.npy
> video_name
> casme_023_0503
> tmp_tf
> Empty DataFrame
> Columns: [subject, video_name, start_frame, apex_frame, end_frame, type_idx, au]
> Index: []
> frame_count
> 2813
> segment for train Finished!
> segment for test Finished!
> ```

在csv文件中没有相关信息，看看数据集中是否有相关信息

源数据集中的xlsx中也没有相关信息，难道是这段视频中没有表情发生？

可能要写一个xlsx和csv转换的代码

casme_023_0503应该是没问题，用一个没有出现在输出中的

这样做有点大海捞针，还是重新测试

在出问题的代码处进行异常抛出和异常捕获

```python
flow_nose_roi = np.stack(flow_nose_roi_list).reshape(-1, 2)
    if flow_nose_roi.size == 0:
        raise ValueError("flow_nose_roi is empty, check ROI boundaries or flow data.")
    flow_nose_roi = get_main_direction_flow(
        flow_nose_roi,
        direction_region=[
            (1 * math.pi / 4, 3 * math.pi / 4),
            (3 * math.pi / 4, 5 * math.pi / 4),
            (5 * math.pi / 4, 7 * math.pi / 4),
            (7 * math.pi / 4, 8 * math.pi / 4, 0, 1 * math.pi / 4),
        ])
    if flow_nose_roi is None:
        raise ValueError("get_main_direction_flow returned None, check flow calculation.")
```

```python
                        try:
                            # 这段可能有问题
                            ior_feature_list = calculate_roi_freature_list(
                                flow_x_y, landmarks, radius=5)
                            ior_feature_list_sequence.append(
                                np.stack(ior_feature_list, axis=0))
                            tq.update()
                        except Exception:
                            ior_feature_list_sequence = []
                            print(f"{sub_item.name}  {type_item.name}")
                            break
```

将之前用于调试的代码改回

进行测试后，在特征提取阶段，出现了这样的问题

> ```python
> casme_030  casme_030_0505
> Traceback (most recent call last):
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_feature.py", line 140, in <module>
>     feature(opt)
>   File "/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/new_feature.py", line 78, in feature
>     flow_x = cv2.imread(flow_x_path_list[i],
> IndexError: list index out of range
> ```

猜测在casme_030_0505中的图片出现了问题，搜索`casme_030_0505.npy`没有出现，因此这个文件夹中的图片处理可能出现了问题

使用这个文件夹进行测试

为防止出现之前023_0503的问题，去csv文件中检测了以下030_0505是存在的

测试代码如下

```python
# 用于调试
!mkdir -p /kaggle/working/rawpic/s30
!cp -r /kaggle/input/casme2/rawpic/rawpic/s30/30_0505funnyinnovations /kaggle/working/rawpic/s30/
```

```python
simpled_root_path: "/kaggle/working/rawpic"
```

出现了错误

> ```bash
> ior_feature_list 有问题
> casme_030  casme_030_0505
> flow_nose_roi is empty, check ROI boundaries or flow data.
> ```

进行边界和流数据的检测

代码如下

```python
        if flow_nose_roi.size == 0:
            print("flow_nose_roi is empty after extraction, checking boundaries...")
            print(
                f"ROI boundaries: top={nose_roi_top}, bottom={nose_roi_bottom}, left={nose_roi_left}, right={nose_roi_right}")
        else:
            print(f"Flow values: min={np.min(flow_nose_roi, axis=0)}, max={np.max(flow_nose_roi, axis=0)}")

```

出现这样的问题

> ```bash
> Flow values: min=[-0.09999999 -0.05294117], max=[-0.09215686 -0.0490196 ]
> flow_nose_roi is empty after extraction, checking boundaries...
> ROI boundaries: top=139, bottom=153, left=34, right=27
> ```

光流的值是正常的，光流值通常应该是较小的浮点数，通常在 -1 到 1 之间波动

但是边界出现了矛盾，left比right大

进行如下修改

```python
# 确保左右边界正确
        if nose_roi_left > nose_roi_right:
            nose_roi_left, nose_roi_right = nose_roi_right, nose_roi_left  # 交换左右边界

        # 确保上下边界正确
        if nose_roi_top > nose_roi_bottom:
            nose_roi_top, nose_roi_bottom = nose_roi_bottom, nose_roi_top  # 交换上下边界

        # 使用np.max和np.min确保ROI边界不越界
        nose_roi_left = np.max([nose_roi_left, 0])
        nose_roi_top = np.max([nose_roi_top, 0])
        nose_roi_right = np.min([nose_roi_right, flows.shape[1] - 1])
        nose_roi_bottom = np.min([nose_roi_bottom, flows.shape[0] - 1])
        # 根据修正后的边界提取ROI
        flow_nose_roi = flows[nose_roi_top:nose_roi_bottom + 1, nose_roi_left:nose_roi_right + 1]
        flow_nose_roi = flow_nose_roi.reshape(-1, 2)
```

这样写就成功运行了

接下来进行整体的运行

将测试代码改回来

```yaml
simpled_root_path: "/kaggle/input/casme2/rawpic/rawpic"
```

```bash
# 用于调试
# !mkdir -p /kaggle/working/rawpic/s30
# !cp -r /kaggle/input/casme2/rawpic/rawpic/s30/30_0505funnyinnovations /kaggle/working/rawpic/s30/
```

数据预处理运行成功了

接着进行模型训练

先上传处理好的数据集

修改提取的特征的地址

```python
segment_feat_root: "/kaggle/input/output/data/casme_2/feature_segment_apex"
```

先下载下预处理的数据特征

#### 预处理误差过大

使用相同的算法但是结果差很多，问题应该出在预处理的过程上

将预处理的到的分割的特征进行对比

> s19 源 45 现 54
>
> s23 源 56 现 47

在进行评估阶段 

使用源项目的特征进行训练的output.csv文件很多，但是自定义提取的非常少。

可能最终的效果就是基于这个进行计算的。

使用源项目的损失率也比较高，甚至有一些比自己提取的还高，但是最终的F1分数却更高

到底是什么原因呢？

先输出各自npz的数据，比较数组中数据的不同点

可能是光流的原因？

或者这个是？

> ```python
> class AUwGCN(torch.nn.Module):
>     def __init__(self, opt):
>         super().__init__()
>         # 这个是？
>         mat_path = os.path.join('/kaggle/working/ME-GCN-Project',
>             'assets',
>             '{}.npy'.format(opt['dataset'])
>         )
> ```

代码中

> ```python
> opt_step = 1  # int(get_micro_expression_average_len(anno_csv_path) // 2)
> ```

但是真正去运行这个函数，结果不为1

结果为

> ```python
> mirco_len = get_micro_expression_average_len("D:/PycharmProjects/ME-GCN-Project/info_csv/cas(me)_new.csv")
> marco_len = get_macro_expression_average_len("D:/PycharmProjects/ME-GCN-Project/info_csv/cas(me)_new.csv")
> print(mirco_len)
> print(marco_len)
> 
> 13.578947368421053
> 40.10546875
> ```

要不改掉这个函数，然后再测试一遍，测试的这一遍不使用

#### 修改参数

对光流的修改

```python
opt_step = int(get_micro_expression_average_len(anno_csv_path) // 2)
cmd = (f'denseflow "{str(type_item)}" -b={opt_step} -a=tvl1 '
                               f'-s={opt_step} -if -o="{new_sub_dir_path}"')
```

对参数的修改

```yaml
dataset: "cas(me)^2"

cas(me)^2:
  dataset: "cas(me)^2"

  # 数据集统计和训练设置
  RATIO_SCALE: 1
  SEGMENT_LENTH: 256  # 视频片段的长度 每个片段被分割成256帧
  RECEPTIVE_FILED: 15  # 感受野 模型一次输入的感受范围
  save_model: True  # 是否保存训练好的模型
  save_intervals: 1   # 模型保存的间隔 每个epoch都保存

  # 微表情和宏表情统计信息
  micro_average_len: 13  # 微表情平均持续长度 帧数
  macro_average_len: 40
  micro_max: 17  # 微表情最大持续长度 帧数
  micro_min: 9
  macro_max: 118
  macro_min: 17  # 实际上计算的是4 第二小是17 但是为了与微表情有一个区别 使用17
  # 为什么epoch_begin 的开始为15
  epoch_begin: 15
  nms_top_K_micro: 5  # 微表情候选数量 每个类别筛选前5个
  nms_top_K_macro: 5

  micro_left_min_dis: 4   # 微表情左侧最小帧数
  micro_left_max_dis: 10  # 微表情左侧最大帧数
  micro_right_min_dis: 4
  micro_right_max_dis: 12

  macro_left_min_dis: 4
  macro_left_max_dis: 63
  macro_right_min_dis: 5  # 宏表情右侧最小帧数
  macro_right_max_dis: 94  # 宏表情右侧最大帧数

  # 路径设置
  project_root: "/kaggle/working/ME-GCN-Project"
  feature_root: ~
  # 源项目特征
  # segment_feat_root: "/kaggle/working/ME-GCN-Project/features/cas(me)^2/feature_segment"
  # 提取完特征不直接开始训练
  # segment_feat_root: "/kaggle/input/output/data/casme_2/feature_segment_apex"
  # 提取完特征 直接开始训练
  segment_feat_root: "/kaggle/working/data/casme_2/feature_segment_apex"
  model_save_root: ~
  output_dir_name: ~
  anno_csv: "/kaggle/working/ME-GCN-Project/info_csv/cas(me)_new.csv"

  # 训练配置
  num_workers: 2  # dataloader 使用的工作线程数
  device: 'cuda:0'

  # ABFCM模型的超参数
  abfcm_training_lr: 0.01  # 学习率
  abfcm_weight_decay: 0.1  # 权重衰减 防止过拟合
  abfcm_lr_scheduler: 0.96  # 学习率调度参数 每个epoch后学习率*0.96
  abfcm_apex_gamma: 1  # 控制apex损失函数相关的参数
  abfcm_apex_alpha: 0.90
  abfcm_action_gamma: 1  # 控制动作相关的损失函数参数
  abfcm_action_alpha: 0.80
  abfcm_start_end_gama: 1  # 控制开始帧和结束帧的损失函数参数
  abfcm_start_end_alpha: 0.90
  abfcm_label_smooth: 0.16  # 标签平滑系数 防止过拟合
  # 为什么是第47个
  abfcm_best_epoch: 47  # 最佳模型在第47个epoch得到 为什么？

  # 分类阈值
  micro_apex_score_threshold: 0.5  # 微表情分类阈值 只有超过0.5才认为该微表情是有效的
  macro_apex_score_threshold: 0.5

  # 训练epoch和批次大小
  epochs: 100
  batch_size: 128

  verbose: False


  macro_ration: 0.5 # 平衡微表情和宏表情的比率 为什么设置成0.5
  micro_normal_range: 1  # 微表情的标准范围 在后处理时的正常偏移量 为什么设置成1
  macro_normal_range: 3

  # 视频数据
  subject_list: [
      "casme_016","casme_015","casme_019","casme_020","casme_021",
      "casme_022","casme_023","casme_024","casme_025","casme_026",
      "casme_027","casme_029","casme_030","casme_031","casme_032",
      "casme_033","casme_034","casme_035","casme_036","casme_037",
      "casme_038","casme_040"
  ]
```

#### 使用已裁剪的图片位置进行裁剪

> ```bash
> FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/facebox_average.csv'
> ```

cropped中没有这个表情的文件夹，可能需要通过其他文件的尺寸进行裁剪，或者使用人脸检测算法进行裁剪

将0101、0401、0402、0502中的平均值写入到0507中

> ```
> /kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/facebox_average.csv不存在
> /kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/facebox_average.csv不存在
> /kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/facebox_average.csv不存在
> /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/facebox_average.csv不存在
> /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/facebox_average.csv不存在
> ```

023怎么到s19下了

> ```
> /kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/facebox_average.csv不存在
> ```

先进行检测 

再进行人脸映射后 检测如下的目录是否缺失 以及有一个原数据集的问题

```python
def check():
    """
    可能是croped 中的路径不存在 但 rawpic的路径存在
    """
    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/ 不存在")

    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/ 不存在")
    # 原数据集的问题
    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/ 不存在")
    # 正常的数据是否存在
    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/ 不存在")

    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/ 不存在")

    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/ 不存在")
```

输出为

> ```
> /kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/ 不存在
> /kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/ 不存在
> /kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/ 不存在
> /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/ 存在
> /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/ 不存在
> /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/ 不存在
> ```

也就是说`s23/casme_023_0502/`这个目录中可能有csv文件

在rawpic的s19中不对023_0502进行操作

对其余四个文件夹进行csv的复制操作 同时检查s23/casme_023_0502下是否有csv文件的存在

s23/casme_023_0502下有csv文件的存在

对`/kaggle/input/casme2/rawpic/rawpic/s19/23_0502funnyerrors`的处理

```python
s_name = "casme_0{}".format(sub_item.name[1:])
v_name = "casme_0{}".format(type_item.name[0:7])
# /kaggle/input/casme2/rawpic/rawpic/s19/23_0502funnyerrors
# 特殊的
if sub_item.name == "s19" and type_item.name == "23_0502funnyerrors":
	s_name = "casme_023"
	v_name = "casme_023_0502"
new_dir_path = os.path.join(
	cropped_root_path, s_name, v_name)
```

```python
if sub_item.name == "s19" and type_item.name == "23_0502funnyerrors":
	facebox_average_path = os.path.join(facebox_csv_root_path, "s23", v_name, "facebox_average.csv")
```

裁剪没有问题，接下来看关键点检测有没有问题

裁剪之后就不做人脸检测

直接使用图片的高宽作为人脸的边界

```python
# 对已经进行人脸裁剪的图像进行检测
# left, top, right, bottom = face_detector.cal(img)
# 已经裁剪过所以不再进行人脸检测
left, top, right, bottom = 0, 0, img.shape[1], img.shape[0]
```

在关键点的检测过程中没有出错

可能要做人脸矫正

有一部分图片剪切的人脸不全在框中，可能有问题

可能要对文件结构进行调整

数据预处理的整个过程所用的时间为7小时15分左右，这样可以把模型的训练放在其中一起进行。省去下载和上传的时间。

利用输出对预处理的代码进行每行注释 调试

处理标注文件的函数放在同一个py文件中

#### 特征可视化对比分析

#### 标签文件修改

原数据集的标签和实际使用的标签不同

对`CAS(ME)^2code_final(Updated).xlsx`中的标注信息进行规范化 用于更好的读取

```python
def get_subject_dict(anno_file):
    """
    输出：
    生成的字典：
    {'1': '15', '2': '16', '3': '19', '4': '20', '5': '21',
        '6': '22', '7': '23', '8': '24', '9': '25', '10': '26',
        '11': '27', '12': '29', '13': '30', '14': '31', '15': '32',
        '16': '33', '17': '34', '18': '35', '19': '36', '20': '37',
        '21': '38', '22': '40'}

    """
    # 读取Excel文件
    xl = pd.ExcelFile(anno_file)
    # 获取第二个工作表
    nameing_rule_1 = xl.parse(xl.sheet_names[1], header=None, dtype=str)
    # 获取第一列的数据并转换为列表
    first_column_list = nameing_rule_1.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    # 获取第三列的数据并转换为列表
    third_column_list = nameing_rule_1.iloc[:, 2].tolist()  # 获取第三列并转换为列表
    subject_dict = {}
    # 检查长度是否一致
    if len(first_column_list) == len(third_column_list):
        # 创建字典，第三列作为键，第一列作为值
        subject_dict = {third_column_list[i]: first_column_list[i] for i in range(len(third_column_list))}
        # 打印字典
        # print("生成的字典：")
        # print(subject_dict)
    else:
        print("第一列和第三列的长度不一致！")

    return subject_dict
```

```python
def get_type_dict(anno_file):
    """
    输出
    生成的字典：
    {'disgust1': '0101', 'disgust2': '0102', 'anger1': '0401',
        'anger2': '0402', 'happy1': '0502', 'happy2': '0503',
        'happy3': '0505', 'happy4': '0507', 'happy5': '0508'}


    """
    # 读取Excel文件
    xl = pd.ExcelFile(anno_file)
    # 获取第三个工作表
    nameing_rule_2 = xl.parse(xl.sheet_names[2], header=None, dtype=str)
    # 获取第一列的数据并转换为列表
    first_column_list = nameing_rule_2.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    # 获取第二列的数据并转换为列表
    second_column_list = nameing_rule_2.iloc[:, 1].tolist()  # 获取第三列并转换为列表
    type_dict = {}
    # 检查长度是否一致
    if len(first_column_list) == len(second_column_list):
        # 创建字典，第二列作为键，第一列作为值
        type_dict = {second_column_list[i]: first_column_list[i] for i in range(len(second_column_list))}
        # 打印字典
        # print("生成的字典：")
        # print(type_dict)
    else:
        print("第一列和第二列的长度不一致！")

    return type_dict
```

```python
def parse_code_final(anno_file):
    """
    首先需要处理
    规范csv文件的格式
    pip install openpyxl
    """
    # 读取subject字典
    subject_dict = get_subject_dict(anno_file)
    # 读取type字典
    type_dict = get_type_dict(anno_file)
    # 读取Excel文件
    xl = pd.ExcelFile(anno_file)
    # 获取第一个工作表
    # 第一行为数据 不是列名
    CASFEcode_final = xl.parse(xl.sheet_names[0], header=None, dtype=str)
    # 获取第一列的数据
    first_column_list = CASFEcode_final.iloc[:, 0].tolist()
    # 遍历第一列，并根据 subject_dict 进行修改
    for idx, item in enumerate(first_column_list):
        # 判断 item 是否在 subject_dict 中
        if item in subject_dict.keys():
            # 替换当前项为格式化的字符串
            first_column_list[idx] = f"casme_0{subject_dict[item]}"
        else:
            print(item)
    # 将修改后的第一列写回到 DataFrame 中
    CASFEcode_final.iloc[:, 0] = first_column_list

    # 获取第二列的数据
    second_column_list = CASFEcode_final.iloc[:, 1].tolist()
    # 遍历第二列，并根据 type_dict 进行修改
    for idx, item in enumerate(second_column_list):
        # 判断 item.split('_')[0] 是否在 type_dict 中
        item = item.split('_')[0]
        if item in type_dict.keys():
            # 替换当前项为格式化的字符串
            second_column_list[idx] = f"{first_column_list[idx]}_{type_dict[item]}"
        else:
            print(item)
    # 将修改后的第二列写回到 DataFrame 中
    CASFEcode_final.iloc[:, 1] = second_column_list

    # 获取第六列的数据
    sixth_column_list = CASFEcode_final.iloc[:, 5].tolist()
    # 将第六列写到第七列 DataFrame 中
    CASFEcode_final.iloc[:, 6] = sixth_column_list

    # 获取第八列的数据
    eighth_column_list = CASFEcode_final.iloc[:, 7].tolist()
    # 遍历第八列，进行替换修改
    for idx, item in enumerate(eighth_column_list):
        if item == "macro-expression":
            eighth_column_list[idx] = "1"
        elif item == "micro-expression":
            eighth_column_list[idx] = "2"
        else:
            print(item)
    # 将修改后的第八列写回到第六列 DataFrame 中
    CASFEcode_final.iloc[:, 5] = eighth_column_list
    # 删除第八列（第7索引）和第九列（第8索引）
    CASFEcode_final.drop(CASFEcode_final.columns[[7, 8]], axis=1, inplace=True)
    # 将列名写入 csv
    col_name_list = ["subject", "video_name", "start_frame", "apex_frame", "end_frame", "type_idx", "au"]
    CASFEcode_final.columns = col_name_list

    if os.path.exists("./cas(me)^2_original.csv"):
        os.remove("./cas(me)^2_original.csv")
    CASFEcode_final.to_csv('./cas(me)^2_original.csv', index=False)
```

使用改csv文件进行特征提取 由于au有空值 可能会报错

由于重新生成csv文件 相应的特征矩阵需要修改

```python
mat_dir = '/kaggle/working/ME-GCN-Project'
# # 本地测试路径
# mat_dir = 'D:/PycharmProjects/ME-GCN-Project'
mat_path = os.path.join(mat_dir, 'assets', '{}_original.npy'.format(opt['dataset']) )
```

宏表情的比率是否要设置成0.8 （从0.84改成0.8）？









还有一种想法，使用现在的人脸裁剪算法，使用之前的san算法

在特征分割中，只有训练集的数据，没有测试集的数据

关于测试集，我的想法是应该用视频进行测试，或者是图片帧。

微表情对检测时间有要求吗

如果用具体的视频进行测试，那么，首先应处理成图片帧，然后进行接下来的处理

如果是训练成模型，那该怎么调用





