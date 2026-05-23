"""结果可视化函数。"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[1]
cache_root = project_root / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """将 BGR 图像转换为 RGB，供 matplotlib 显示。"""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def create_summary_figure(
    original_bgr: np.ndarray,
    downsampled_bgr: np.ndarray,
    bicubic_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    mask_bgr: np.ndarray,
    segmentation_bgr: np.ndarray,
    output_path: Path,
) -> None:
    """绘制 2x3 的对比子图。"""
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))

    items = [
        ("Original Image", original_bgr),
        ("Downsampled Image", downsampled_bgr),
        ("Bicubic Result", bicubic_bgr),
        ("GFPGAN Result", restored_bgr),
        ("Segmentation Mask", mask_bgr),
        ("Segmentation Result", segmentation_bgr),
    ]

    for ax, (title, image_bgr) in zip(axes.ravel(), items):
        ax.imshow(bgr_to_rgb(image_bgr))
        ax.set_title(title)
        ax.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
