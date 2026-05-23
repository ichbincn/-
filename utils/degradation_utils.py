"""图像退化相关函数。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.io_utils import save_image


def create_downsampled_image(original_bgr: np.ndarray, output_dir: Path) -> dict[str, np.ndarray]:
    """
    仅通过下采样构造低分辨率图像。

    说明：
    - 这里不再额外加入高斯噪声、椒盐噪声或模糊。
    - 宽和高均缩小为原图的 1/4，便于后续 4x ESRGAN 恢复。
    """
    height, width = original_bgr.shape[:2]
    low_width = max(1, width // 4)
    low_height = max(1, height // 4)

    downsampled_lr = cv2.resize(
        original_bgr,
        (low_width, low_height),
        interpolation=cv2.INTER_AREA,
    )
    save_image(output_dir / "downsampled_lr.png", downsampled_lr)

    downsampled_display = cv2.resize(
        downsampled_lr,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    save_image(output_dir / "downsampled_display.png", downsampled_display)

    return {
        "downsampled_lr": downsampled_lr,
        "downsampled_display": downsampled_display,
    }
