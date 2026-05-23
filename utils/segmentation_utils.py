"""人像分割相关函数。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.io_utils import save_image


def run_grabcut_segmentation(image_bgr: np.ndarray, output_dir: Path) -> dict[str, np.ndarray]:
    """
    使用 GrabCut 对人像进行简单分割。

    这里假设人像大致位于图像中心区域，因此使用中心矩形作为初始化区域。
    该方法简单稳定，适合课程设计演示。
    """
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), np.uint8)

    # 中心矩形：保留四周少量边距，作为 GrabCut 初始前景区域
    rect = (
        int(width * 0.1),
        int(height * 0.05),
        int(width * 0.8),
        int(height * 0.9),
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_bgr,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as exc:
        raise RuntimeError(f"GrabCut 分割失败: {exc}") from exc

    binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    foreground = cv2.bitwise_and(image_bgr, image_bgr, mask=binary_mask)

    save_image(output_dir / "segmentation_mask.png", binary_mask)
    save_image(output_dir / "segmentation_foreground.png", foreground)

    return {
        "mask": binary_mask,
        "foreground": foreground,
    }
