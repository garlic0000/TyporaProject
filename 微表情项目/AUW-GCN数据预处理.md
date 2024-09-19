# AUW-GCN数据预处理

## CAS(ME)^2

### 实验

#### 直接使用数据集裁剪好的图片

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

#### 使用数据集选取的关键帧进行统一裁剪

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

接下来将右侧进行裁剪













关于命名，应该在最后进行特征分割的时候，在保存时进行重命名，需要标注文件xlsx和csv的转换

关于测试集，我的想法是应该用视频进行测试，或者是图片帧。

微表情对检测时间有要求吗

如果用具体的视频进行测试，那么，首先应处理成图片帧，然后进行接下来的处理

如果是训练成模型，那该怎么调用









#### 直接使用数据集的关键帧作为已裁剪的图片

这个可能不可行，因为没有人脸和关键点的位置文件

现在的问题时解决检测人脸和人脸关键点的问题。

但是可以进行人脸检测和关键点检测的代码有无问题。

使用新的算法还是有问题

打算使用未裁剪的图片进行人脸检测和关键点检测

可能是`CROPPED_SIZE`设置的有问题，或者填充算法设置的有问题，导致检测不到人脸

或者像素有问题

这个选项已经不需要进行测试

#### 使用之前的人脸检测和关键点检测算法



### 取样

抽取关键帧，即表情帧

但是这个数据集中自带关键帧文件夹，CAS(ME)^2数据集中的selectpic这个文件夹中为研究人员选择的表情帧，即出现宏表情和微表情时间片段中组成的帧。

### 裁剪

数据集自带裁剪好的图片，但是这个图片裁剪的让人脸检测软件和关键点检测软件检测不出，无法继续进行。

只能使用未修剪的表情帧统一修剪，然后进行关键点标记，不知道之后是否会影响检测效果。

### 人脸检测和关键点标记

源代码中使用以下两个模型进行人脸检测和人脸关键点检测

> ```python
> face_det_model_path = "/kaggle/input/checkpoint/pytorch/default/1/retinaface_Resnet50_Final.pth"
> face_detector = FaceDetector(face_det_model_path)
> landmark_model_path = '/kaggle/input/checkpoint/pytorch/default/1/san_checkpoint_49.pth.tar'
> landmark_detector = LandmarkDetector(landmark_model_path)
> ```



