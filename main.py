"""
基于人脸恢复与图像分割的人像图像处理系统

面向课程设计选题一的简化实现：
1. 读取原图
2. 仅做下采样退化
3. 使用 Bicubic interpolation 恢复图像尺寸
4. 使用 GFPGAN 做人脸恢复增强
5. 使用 GrabCut 做人像分割
6. 计算 PSNR / SSIM
7. 生成结果可视化图
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from utils.degradation_utils import create_downsampled_image
from utils.enhancement_utils import bicubic_restore_image
from utils.face_restoration_utils import run_gfpgan_face_restoration
from utils.io_utils import ensure_directories, read_image, save_image
from utils.metrics_utils import evaluate_and_print_metrics
from utils.segmentation_utils import run_grabcut_segmentation
from utils.visualization_utils import create_summary_figure


def main() -> None:
    project_root = Path(__file__).resolve().parent
    candidate_inputs = [
        project_root / "fig.jpg",
        project_root / "fig.png",
        project_root / "fig.jpeg",
    ]
    input_path = next((path for path in candidate_inputs if path.exists()), None)
    output_dir = project_root / "outputs"
    model_dir = project_root / "models"

    np.random.seed(42)
    ensure_directories([output_dir, model_dir])

    if input_path is None:
        raise FileNotFoundError(
            "未找到输入图像，请在项目根目录放置 fig.jpg、fig.png 或 fig.jpeg"
        )

    print("=" * 70)
    print("基于人脸恢复与图像分割的人像图像处理系统")
    print("三种核心算法：Bicubic + GFPGAN + GrabCut")
    print("=" * 70)

    # STEP 1：读取原图
    original_bgr = read_image(input_path)
    save_image(output_dir / "original_image.png", original_bgr)
    print("[1/7] 原始图像读取并保存完成。")

    # STEP 2：仅做下采样退化
    downsample_results = create_downsampled_image(original_bgr, output_dir)
    downsampled_lr_bgr = downsample_results["downsampled_lr"]
    downsampled_display_bgr = downsample_results["downsampled_display"]
    print("[2/7] 下采样退化完成。")

    # STEP 3：使用 Bicubic 恢复图像尺寸
    bicubic_bgr = bicubic_restore_image(
        downsampled_lr_bgr=downsampled_lr_bgr,
        target_size=(original_bgr.shape[1], original_bgr.shape[0]),
        output_dir=output_dir,
    )
    print("[3/7] Bicubic 图像恢复完成。")

    # STEP 4：使用 GFPGAN 做人脸恢复增强
    gfpgan_bgr = run_gfpgan_face_restoration(
        input_bgr=bicubic_bgr,
        model_dir=model_dir,
        output_dir=output_dir,
    )
    print("[4/7] GFPGAN 人脸恢复完成。")

    # STEP 5：人像分割
    segmentation_results = run_grabcut_segmentation(gfpgan_bgr, output_dir)
    segmented_foreground_bgr = segmentation_results["foreground"]
    segmentation_mask = segmentation_results["mask"]
    segmentation_mask_bgr = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
    print("[5/7] 人像分割完成。")

    # STEP 6：图像质量评价
    metrics = evaluate_and_print_metrics(
        original_bgr=original_bgr,
        downsampled_bgr=downsampled_display_bgr,
        bicubic_bgr=bicubic_bgr,
        restored_bgr=gfpgan_bgr,
        output_path=output_dir / "metrics.txt",
    )
    print("[6/7] 图像质量评价完成。")

    # STEP 7：结果可视化
    create_summary_figure(
        original_bgr=original_bgr,
        downsampled_bgr=downsampled_display_bgr,
        bicubic_bgr=bicubic_bgr,
        restored_bgr=gfpgan_bgr,
        mask_bgr=segmentation_mask_bgr,
        segmentation_bgr=segmented_foreground_bgr,
        output_path=output_dir / "comparison_figure.png",
    )
    print("[7/7] 对比可视化完成。")

    print("-" * 70)
    print("PSNR / SSIM 结果摘要：")
    for name, values in metrics.items():
        print(
            f"{name:>10s} -> PSNR: {values['psnr']:.4f} dB, "
            f"SSIM: {values['ssim']:.4f}"
        )
    print("-" * 70)
    print(f"全部结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
