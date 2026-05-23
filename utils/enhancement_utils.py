"""低分辨率图像恢复相关函数。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.io_utils import save_image


def bicubic_restore_image(
    downsampled_lr_bgr: np.ndarray,
    target_size: tuple[int, int],
    output_dir: Path,
) -> np.ndarray:
    """使用 Bicubic 插值将低分辨率图像恢复到原始尺寸。"""
    bicubic_bgr = cv2.resize(
        downsampled_lr_bgr,
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    save_image(output_dir / "bicubic_result.png", bicubic_bgr)
    return bicubic_bgr
