# 目标检测基础知识

## 卷积层面

### 空洞卷积（Dilated Convolution）

空洞卷积也叫扩张卷积或膨胀卷积，简单来说就是在卷积核元素之间加入一些空格来扩大卷积核的过程。与常规卷积相比具备更大的感受野。

![空洞卷积](/assets/image/1.png)

建设以变量a来衡量空洞卷积的扩张系数，则加入空洞之后的实际卷积核尺寸与原始卷积核尺寸之间的关系：K = K + (k-1)(a-1)。其实感受野还有一点比较重要的是，对于一个卷积特征图而言，感受野中每个像素并不是同等重要的，越接近感受野中间的像素相对而言就越重要。

空洞卷积主要有三个作用：
* 扩大感受野。但需要明确一点，池化也可以扩大感受野，但空间分辨率降低了，相比之下，空洞卷积可以在扩大感受野的同时不丢失分辨率，且保持像素的相对空间位置不变。简单而言，空洞卷积可以同时控制感受野和分辨率。
* 获取多尺度上下文信息。当多个带有不同dilation rate的空洞卷积核叠加时，不同的感受野会带来多尺度信息，这对于分割任务是非常重要的。
* 可以降低计算量，不需要引入额外的参数，如上图空洞卷积示意图所示，实际卷积时只有带有红点的元素真正进行计算。