"""传统数字图像处理算法实现。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.io_utils import save_image


def histogram_equalization_color(image_bgr: np.ndarray) -> np.ndarray:
    """对彩色图像的亮度通道做直方图均衡化。"""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    equalized_y = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([equalized_y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def fourier_spectrum_visualization(gray_image: np.ndarray) -> np.ndarray:
    """生成傅里叶频谱可视化图。"""
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log(np.abs(f_shift) + 1.0)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def high_pass_filter_frequency(gray_image: np.ndarray) -> np.ndarray:
    """使用频域高通滤波进行高频增强。"""
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2

    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)

    # 构造简单高通掩模：中心低频区域置零，其余保留
    mask = np.ones((rows, cols), dtype=np.float32)
    radius = min(rows, cols) // 10
    y, x = np.ogrid[:rows, :cols]
    center_region = (y - crow) ** 2 + (x - ccol) ** 2 <= radius**2
    mask[center_region] = 0.0

    filtered_shift = dft_shift * mask
    restored = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    restored = np.abs(restored)
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)
    return restored.astype(np.uint8)


def run_classical_processing(image_bgr: np.ndarray, output_dir: Path) -> None:
    """执行所有传统图像处理并保存结果。"""
    hist_eq = histogram_equalization_color(image_bgr)
    save_image(output_dir / "traditional_histogram_equalization.png", hist_eq)

    gaussian_filtered = cv2.GaussianBlur(image_bgr, (5, 5), sigmaX=1.2)
    save_image(output_dir / "traditional_gaussian_filter.png", gaussian_filtered)

    median_filtered = cv2.medianBlur(image_bgr, 5)
    save_image(output_dir / "traditional_median_filter.png", median_filtered)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    spectrum = fourier_spectrum_visualization(gray)
    save_image(output_dir / "traditional_fourier_spectrum.png", spectrum)

    high_pass = high_pass_filter_frequency(gray)
    save_image(output_dir / "traditional_high_pass.png", high_pass)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    save_image(output_dir / "traditional_sobel_edge.png", sobel_magnitude.astype(np.uint8))

    canny_edges = cv2.Canny(gray, 80, 160)
    save_image(output_dir / "traditional_canny_edge.png", canny_edges)
