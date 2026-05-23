"""图像质量评价函数。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(reference_bgr: np.ndarray, target_bgr: np.ndarray) -> dict[str, float]:
    """计算 PSNR 和 SSIM。"""
    reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    psnr_value = peak_signal_noise_ratio(reference_rgb, target_rgb, data_range=255)
    ssim_value = structural_similarity(
        reference_rgb,
        target_rgb,
        channel_axis=2,
        data_range=255,
    )
    return {
        "psnr": float(psnr_value),
        "ssim": float(ssim_value),
    }


def evaluate_and_print_metrics(
    original_bgr: np.ndarray,
    downsampled_bgr: np.ndarray,
    bicubic_bgr: np.ndarray,
    restored_bgr: np.ndarray,
    output_path: Path,
) -> dict[str, dict[str, float]]:
    """计算并打印所有评价指标，同时保存到文本文件。"""
    metrics = {
        "downsampled": compute_metrics(original_bgr, downsampled_bgr),
        "bicubic": compute_metrics(original_bgr, bicubic_bgr),
        "gfpgan": compute_metrics(original_bgr, restored_bgr),
    }

    lines = [
        "图像质量评价结果（参考图像：original_image.png）",
        "-" * 50,
    ]
    for name, values in metrics.items():
        line = (
            f"{name:>10s} -> "
            f"PSNR: {values['psnr']:.4f} dB, "
            f"SSIM: {values['ssim']:.4f}"
        )
        lines.append(line)

    print("\n".join(lines))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
