"""人脸恢复模型相关函数。"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import setuptools._distutils as _distutils
import torch
import torchvision.transforms._functional_tensor as _functional_tensor

from utils.io_utils import save_image


GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)


def patch_gfpgan_compatibility() -> None:
    """
    兼容 Python 3.12 与当前 torchvision 版本。

    GFPGAN 依赖链中的部分老版本库仍引用：
    - distutils
    - torchvision.transforms.functional_tensor

    这里在运行时做一次轻量兼容映射，保证项目可直接运行。
    """
    sys.modules.setdefault("distutils", _distutils)
    sys.modules.setdefault(
        "torchvision.transforms.functional_tensor",
        _functional_tensor,
    )


def download_gfpgan_model_if_needed(model_path: Path) -> None:
    """如果 GFPGAN 权重不存在，则自动下载。"""
    if model_path.exists():
        return

    try:
        print(f"正在下载 GFPGAN 预训练模型: {GFPGAN_MODEL_URL}", flush=True)
        urlretrieve(GFPGAN_MODEL_URL, str(model_path))
        print(f"GFPGAN 模型下载完成: {model_path}", flush=True)
    except (URLError, OSError, RuntimeError) as exc:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(
            "GFPGAN 模型下载失败。请检查网络连接，或手动将 "
            f"GFPGANv1.4.pth 放入目录: {model_path.parent}\n"
            f"错误信息: {exc}"
        ) from exc


def run_gfpgan_face_restoration(
    input_bgr: cv2.typing.MatLike,
    model_dir: Path,
    output_dir: Path,
) -> cv2.typing.MatLike:
    """
    使用 GFPGAN 对人脸图像做恢复增强。

    处理策略：
    1. 先对低分辨率图像做 Bicubic 放大
    2. 再使用 GFPGAN 仅对中心人脸区域进行恢复
    3. 将恢复后的人脸粘贴回整张图像
    """
    patch_gfpgan_compatibility()
    from gfpgan import GFPGANer

    model_path = model_dir / "GFPGANv1.4.pth"
    download_gfpgan_model_if_needed(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GFPGAN 推理设备: {device}")

    try:
        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        _, _, restored_bgr = restorer.enhance(
            input_bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            weight=0.6,
        )
    except Exception as exc:
        raise RuntimeError(f"GFPGAN 推理失败: {exc}") from exc

    if restored_bgr is None:
        raise RuntimeError("GFPGAN 未返回恢复结果，请检查输入图像或模型文件。")

    save_image(output_dir / "gfpgan_result.png", restored_bgr)
    return restored_bgr
