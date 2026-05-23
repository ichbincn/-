"""输入输出相关工具函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def ensure_directories(paths: Iterable[Path]) -> None:
    """确保目录存在，不存在则自动创建。"""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_image(image_path: Path) -> np.ndarray:
    """读取图像并检查是否成功。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    return image


def save_image(image_path: Path, image: np.ndarray) -> None:
    """保存图像到指定路径。"""
    success = cv2.imwrite(str(image_path), image)
    if not success:
        raise ValueError(f"图像保存失败: {image_path}")
