# Yolov5在Ubuntu环境搭建

## 1. 系统环境配置

### 1.1 显卡驱动安装

通过自带的Software & Updates工具通过如下界面进行驱动安装。
![驱动](/assets/image/2.jpg)
安装完成后重启，然后通过nvidia-smi查看是否安装成功。

### 1.2 CUDA安装

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```

接着需要配置对应的环境变量，对应指令如下所示。

```bash
sudo vi ~/.bashrc #增加如下

export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lcoal/cuda-11.3/lib64:$LD_LIBRARY_PATH"
```

通过指令source ~/.bashrc使其生效，完成安装后我们还需要进行测试，确保其CUDA可用，这里我们利用示例项目进行测试。

```bash
cd /usr/local/cuda-11.3/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

也可以通过nvcc -V查看具体情况。

### 1.3 cuDNN安装
由于本安装包需要提前进行下载，无法通过免登进行下载，所以需要通过本[网址](https://developer.nvidia.com/rdp/cudnn-archive)选择对应支持的版本进行下载，然后将包进行上传，通过如下指令进行安装。

```bash
sudo tar -xzvf cudnn-11.3-linux-x64-v8.2.1.32.tgz

sudo cp cuda/include/cudnn* /usr/local/cuda-11.3/include/
sudo chmod a+r /usr/local/cuda-11.3/include/cudnn*
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64/
sudo chmod a+r /usr/local/cuda-11.3/lib64/libcudnn*
```

最后通过如下指令查看版本确认是否已安装成功。

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

## 2. Python环境配置

### 2.1 Anaconda安装
首先需要通过清华源下载对应的安装脚本文件，然后进行安装。

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2022.05-Linux-x86_64.sh

bash Anaconda3-2022.05-Linux-x86_64.sh
```

接着修改环境变量以便能够启用。

```bash
sudo vi ~/.bashrc

export PATH="/home/yzf/anaconda3/bin:$PATH"
```

使用source ~/.bashrc进行生效，接着通过conda –version查看是否安装成功。最后就是配置清华源，以便加速安装包下载。

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

最后可以通过conda config --show channels查看具体的通道是否已添加。

### 2.2 pip换源
由于默认的Pip源为国外源，在实际使用过程中下载对应安装包速度较慢，为此我们需要切换对应的源，具体操作如下。

```bash
mkdir ~/.pip
vim ~/.pip/pip.conf
```

接着在文件中写入如下源，下述为阿里源。

```conf
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host = mirrors.aliyun.com
```

对应备用的清华源。
```conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
```
完成上述的源后需要对pip进行更新，执行pip install update即可。

### 2.3 git安装

下面主要以安装以及秘钥配置为主进行介绍。

```bash
sudo apt install git
sudo touch /usr/.ssh
ssh-keygen -t rsa -C "xxx@xxx.com" -b 4096
```
具体公钥见id_rsa.pub文件中的内容进行添加即可。

## 3. yolov5框架搭建

### 3.1 模型项目
首先将项目源码进行下载，对应在配置具体的环境参数，具体如下所示。

```bash
git clone https://github.com/ultralytics/yolov5.git

conda create -n yolov5 python=3.8
conda activate yolov5
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
```

### 3.2 模型提速
由于yolov5模型自4.0版本开始激活函数已经从LeakyReLU调整为SiLU，这将导致在使用新版后模型在地平线芯片上实际执行的效能无法满足15ms，为此我们需要调整models/common.py文件中对应位置的函数，下述根据不同版本进行介绍具体修改位置。

* v6.0-v5.0-v4.0版本
![3](/assets/image/3.jpg)

此处调整为如下代码

```python
self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
```

* v6.1-v6.2版本
![4](/assets/image/4.jpg)

此处调整为如下代码

```python
self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
```

![5](/assets/image/5.jpg)

此处调整为如下代码
```python
self.act = nn.LeakyReLU(0.1, inplace=True)
```

### 3.3 模型训练
由于模型需要进行转换以支持最终的边缘终端设备，需要规范化其训练的过程参数，其中模型转换主要支持的是640与512尺寸的输入。所以完成对应的训练脚本示例如下所示：

```bash
python train.py --img 640 --batch-size 16 --epochs 100 --cfg models/yolov5s.yaml --data data.yaml --weights yolov5n.pt --name vest
```

以上指令中对应的参数说明如下：
* img：代表图片输入尺寸，主要为640、512与256；
* batch-size：批大小，即每批样本的大小，一般根据GPU内存大小决定；
* epochs：迭代数，代表完整的数据集需要进行多少次迭代；
* data：加载代表数据的配置项；
* weights：代表使用的权重文件，一般都是采用已训练好的权重文件；
* name：指定最终生成至runs的train下的文件夹名称；

### 3.4 模型选择
根据平台所具备的算力需要选择合适的模型满足具体的需求，比如基于地平线芯片且输出二阶段识别则推荐使用yolov5n进行训练并且输入的图片尺寸建议使用256或128输入。仅针对部分单阶段的模型采用yolov5s进行训练，具体的对应的模型以及运算复杂程度如下图。

![6](/assets/image/6.jpg)

### 3.5 模型验证
针对已完成训练的模型，如果希望通过其他外部图片或视频进行验证可以使用以下方式进行验证。

```bash
python detect.py --source data/images --weights runs\train\exp\weights\best.pt --conf-thres 0.25
```

其中该方式的调用更多的参数说明如下。

* weights：需要验证的权重文件；
* source：测试数据，可以是图片、视频路径，也可以是’0’（电脑自带摄像头）或RTSP视频流地址；
* output：网络预测之后的图片/视频的保存路径；
* img-size：网络输入图片大小；
* conf-thres：置信度阈值；
* iou-thres：做nms的iou阈值；
* device：进行推理的显卡设备序号；
* view-img：是否展示预测之后的图片/视频，默认False；
* save-txt：是否将预测的框坐标以txt文件形式保存，默认False；
* classes：设置只保留某一部分类别，行为0或0 2 3；
* agnostic-nms：进行nm是否也去除不同类别之间的框，默认False；
* augment：推理的时候进行多尺度，翻转等操作推理；
* update：如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认False；

### 3.6 模型导出
由于默认训练模型的依赖未包含onnx库，首先需要通过conda切换到对应的环境中进行库安装，然后利用yolov5的export.py进行导出即可，具体指令如下：

```bash
conda activate yolov5
pip install onnx
python export.py –weights [权重文件.pt] --img 640 --batch 1
```

此时可以看到对应的模型输出，如果需要查看onnx的网络结构可以通过netron进行查看，具体的使用方式如下：

```bash
pip install onnx

python>>>

import netron
netron.start('yolov5s.onnx')
```
简化onnx文件
```bash
pip install onnx-simplifier
python -m onnxsim  onnx_inputpath onnx_outputpath
```

## 4. yolov5-lite框架搭建

### 4.1 模型项目
首相将项目源码进行下载，对应在配置具体的环境参数，具体如下所示。

```bash
git clone https://github.com/ppogg/YOLOv5-Lite

conda create -n yolov5-lite python=3.9
conda activate yolov5-lite
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### 4.2 模型训练

其中用法与传统yolov5类似。
```bash
python train.py --data data/vest_1/vest_1.yaml --cfg models/v5Lite-e.yaml --weights v5lite-e.pt --batch-size 32 --epochs 200 --name vest_1
```

### 4.3 模型验证

```bash
python detect.py --source file.mp4 --weights runs\train\exp\weights\best.pt --conf-thres 0.25
```
最后的结果将保存至runs/detect中。

### 4.3 模型导出
如果需要导出其他各类型的模型可以使用下述方式进行导出。

```bash
conda activate yolov5-lite
pip install onnx
python export.py --weights [权重文件.pt] --img 640 --batch 1
```

## 5. yolov5-face框架搭建
由于存在人脸抓拍需求，当前框架主要基于开源的yolov5-face进行训练，其具备高识别率以及可训练的特点，从而能够满足各类场景下的人脸抓拍诉求。

### 5.1 模型项目
首先需要将对应项目进行下载，为了便于进行模型训练，首先需要将对应项目通过git进行下载，具体指令如下：

```bash
git clone https://github.com/deepcam-cn/yolov5-face.git
conda create -n yolov5-face python=3.8
conda activate yolov5-face
pip install -r requirements.txt
```

由于其需要特殊的素材进行训练需要通过http://shuoyang1213.me/WIDERFACE/下载对应的素材文件，主要由以下几个文件构成：

* WIDER_val.zip
* WIDER_train.zip
* WIDER_test.zip
其中由于标注数据的特殊性，需要通过云盘进行下载。
```
链接：https://pan.baidu.com/s/1RaO-MCfldBDnVlkXkKCk8Q?pwd=1nph 
提取码：1nph
```

将上述素材解压到yolov5-face项目的data/widerface-source目录下，同时将云盘中的标注数据分别放置在对应文件夹（如WIDER_train）目录下，对应执行如下指令。

```bash
python train2yolo.py /path/to/original/widerface/train
python val2yolo.py /path/to/original/widerface/val
```

目录需要可读取到对应的label.txt文件，同时由于素材的数量较多，转换需要花费一定的时间。默认输出为脚本当前目录下的widerface文件夹中。

`注意：其中val2yolo.py需要调整第64行与78行代码，其自行增加了val文件夹，将其删除。`

### 5.2 移植适配
![7](/assets/image/7.jpg)
打开对应的common.py文件，将其中第43行注释后修改为第44行代码即可。

### 5.3 模型训练
修改data/widerface.yml文件内容，重新定位数据集。
```yaml
train: ../widerface/train
val: ../widerface/val

# number of classes
nc: 1

# class names
names: [ 'face']
```
在实际训练前除了yolov5必要的依赖以外，此项目还需要其他额外的引用，这里通过conda进行下载引用，具体如下。
```bash
conda install PyYAML tqdm tensorboard matplotlib pandas seaborn onnx
pip install thop
```
完成数据集的相关准备工作后，我们通过下述指令开始进行模型训练。
```bash
python train.py --weights=./yolov5s.pt --data=./data/widerface.yaml --epochs=100 --batch-size=8 --device=0
```