# 基于人脸恢复与图像分割的人像图像处理系统

本项目面向《数字图像处理》课程设计 `选题一（算法设计类）`，围绕“下采样退化后的人像图像增强与分割”问题，构建了一套可直接运行的完整 Python 项目。

当前最终版主流程只保留 3 个核心算法：

1. `Bicubic Interpolation`
2. `GFPGAN`
3. `GrabCut`

说明：
- 下采样仅作为退化步骤，不计入“三个核心算法”。
- 该版本已经替换掉之前效果不理想的通用 ESRGAN 方案，改为更适合人脸图像增强的专用恢复模型 `GFPGAN`。

## 项目结构

```text
project/
│
├── fig.jpg / fig.png / fig.jpeg
├── main.py
├── requirements.txt
├── README.md
├── 创新点与90分思路.md
├── outputs/
├── models/
└── utils/
```

## 环境建议

推荐使用：
- Python 3.12

说明：
- 项目内已经对 `GFPGAN` 在 Python 3.12 下的兼容性做了运行时补丁处理，因此无需手工改第三方源码。

## 安装依赖

```bash
pip install -r requirements.txt
```

如果直接使用当前项目已经建立好的虚拟环境，运行方式更简单：

```bash
.venv/bin/python main.py
```

## 输入文件

程序会自动识别以下输入文件名之一：
- `fig.jpg`
- `fig.png`
- `fig.jpeg`

## 主流程说明

### STEP 1：读取原图

读取输入人像图像并保存：
- `outputs/original_image.png`

### STEP 2：下采样退化

只执行分辨率降低，不再额外添加高斯噪声、椒盐噪声和模糊。程序将原图缩小为原始宽高的 `1/4`，得到低分辨率图像：
- `outputs/downsampled_lr.png`
- `outputs/downsampled_display.png`

### STEP 3：Bicubic 图像恢复

使用 `Bicubic Interpolation` 将低分辨率图像恢复到原图尺寸：
- `outputs/bicubic_result.png`

### STEP 4：GFPGAN 人脸恢复

对 Bicubic 恢复结果进一步使用 `GFPGAN` 进行人脸增强。

该步骤特点：
- 专门面向人脸图像恢复
- 自动检测中心人脸
- 对五官与面部结构做细节增强
- 将增强后的人脸粘贴回整张图像

输出文件：
- `outputs/gfpgan_result.png`

模型文件：
- `models/GFPGANv1.4.pth`

说明：
- 首次运行会自动下载 GFPGAN 模型。
- 人脸检测与解析辅助权重会自动下载到项目目录下的 `gfpgan/weights/`。
- 如果下载失败，程序会给出明确错误提示。

### STEP 5：GrabCut 人像分割

对增强后的人像图像使用 `GrabCut` 进行前景提取，生成：
- `outputs/segmentation_mask.png`
- `outputs/segmentation_foreground.png`

### STEP 6：图像质量评价

以原图为参考，计算：
- `PSNR`
- `SSIM`

比较对象包括：
- `downsampled`
- `bicubic`
- `gfpgan`

结果保存到：
- `outputs/metrics.txt`

### STEP 7：结果可视化

程序会自动生成：
- `outputs/comparison_figure.png`

其中包含：
- Original Image
- Downsampled Image
- Bicubic Result
- GFPGAN Result
- Segmentation Mask
- Segmentation Result

## 三个核心算法原理

### 1. Bicubic Interpolation

双三次插值是一种经典图像缩放方法。它通过周围多个像素共同估计目标像素值，相较于最近邻插值和双线性插值，能够得到更平滑、更自然的缩放结果，因此适合作为低分辨率图像的基础恢复步骤。

### 2. GFPGAN

GFPGAN（Generative Facial Prior GAN）是一种专门面向人脸恢复任务的深度学习模型。它利用生成式先验信息增强人脸结构表达能力，能够对模糊、低清或退化后的人脸图像进行更自然的细节恢复。

在本项目中，GFPGAN 的作用是：
- 提升五官清晰度
- 改善面部边缘与轮廓
- 让恢复结果更符合人脸视觉规律

### 3. GrabCut

GrabCut 是一种经典图像分割算法。它通过建立前景与背景的颜色模型，并结合图割优化过程，不断迭代更新分割边界。对于单人正面证件照一类图像，GrabCut 往往能够取得较稳定的分割结果。

## 当前实测结果

本项目已经在本地端到端验证通过，当前评价结果如下：

```text
downsampled -> PSNR: 26.2078 dB, SSIM: 0.7885
bicubic     -> PSNR: 27.3906 dB, SSIM: 0.8119
gfpgan      -> PSNR: 27.6409 dB, SSIM: 0.8147
```

从结果可以看到，`GFPGAN` 相比单独 `Bicubic` 恢复进一步提升了客观指标，同时在视觉上也更符合人脸增强任务需求。

## 适合课程设计报告的创新点

项目已经额外整理出一个专门文档：

- [创新点与90分思路.md](/Users/chennuo/Desktop/图像proj/创新点与90分思路.md)

其中总结了 3 个适合写进正文的创新点：
- 使用人脸专用恢复模型替代通用超分模型
- 采用“Bicubic 预恢复 + GFPGAN 人脸增强”的两阶段恢复策略
- 将增强结果与分割任务联动，提高前景提取效果

## 推荐查看的最终结果文件

- [comparison_figure.png](/Users/chennuo/Desktop/图像proj/outputs/comparison_figure.png)
- [gfpgan_result.png](/Users/chennuo/Desktop/图像proj/outputs/gfpgan_result.png)
- [segmentation_foreground.png](/Users/chennuo/Desktop/图像proj/outputs/segmentation_foreground.png)
- [metrics.txt](/Users/chennuo/Desktop/图像proj/outputs/metrics.txt)

## 备注

- `outputs/` 已经清理过，只保留当前最终方案需要的结果图。
- 项目里仍保留部分早期实验代码模块，便于你在报告中说明方案对比与迭代过程，但它们不再参与默认主流程。
