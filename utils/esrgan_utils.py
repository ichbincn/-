"""ESRGAN / Real-ESRGAN 超分辨率相关函数。"""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils.io_utils import save_image


class ResidualDenseBlock_5C(nn.Module):
    """ESRGAN 中的 5 层残差稠密块。"""

    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    """Residual in Residual Dense Block。"""

    def __init__(self, nf: int, gc: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class RRDBNet(nn.Module):
    """兼容 Real-ESRGAN x4plus 权重命名的 RRDBNet 结构。"""

    def __init__(
        self,
        in_nc: int,
        out_nc: int,
        nf: int,
        nb: int,
        gc: int = 32,
    ) -> None:
        super().__init__()
        rrdb_blocks = [RRDB(nf, gc=gc) for _ in range(nb)]

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*rrdb_blocks)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        body_feat = self.conv_body(self.body(fea))
        fea = fea + body_feat

        fea = self.lrelu(
            self.conv_up1(
                nn.functional.interpolate(fea, scale_factor=2, mode="nearest")
            )
        )
        fea = self.lrelu(
            self.conv_up2(
                nn.functional.interpolate(fea, scale_factor=2, mode="nearest")
            )
        )
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return out


def download_model_if_needed(model_path: Path) -> None:
    """
    若模型不存在则自动下载。

    下载更适合真实人像图像的 Real-ESRGAN x4plus 权重。
    它是 ESRGAN 系列的实用改进版本，视觉效果通常比原始 ESRGAN 更自然。
    """
    if model_path.exists():
        return

    candidate_urls = [
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ]

    last_error: Exception | None = None
    for url in candidate_urls:
        try:
            print(f"正在下载 ESRGAN 预训练模型: {url}", flush=True)
            urlretrieve(url, str(model_path))
            print(f"模型下载完成: {model_path}", flush=True)
            return
        except (URLError, OSError, RuntimeError) as exc:
            last_error = exc
            if model_path.exists():
                model_path.unlink()
            print(f"模型下载失败，尝试下一个地址: {exc}", flush=True)

    raise RuntimeError(
        "ESRGAN 模型下载失败。请检查网络连接，或手动将 "
        f"RealESRGAN_x4plus.pth 放入目录: {model_path.parent}\n"
        f"最后一次错误信息: {last_error}"
    )


def load_esrgan_model(model_path: Path, device: torch.device) -> RRDBNet:
    """加载 ESRGAN 预训练模型。"""
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 兼容带 module. 前缀的权重键名
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def run_esrgan_inference(
    image_bgr: np.ndarray,
    model: RRDBNet,
    device: torch.device,
) -> np.ndarray:
    """使用 ESRGAN 对单张低分辨率图像执行 4x 超分。"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = (
        torch.from_numpy(np.transpose(image_rgb, (2, 0, 1)))
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        output = model(tensor).clamp(0.0, 1.0)

    output_np = output.squeeze(0).cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))
    output_np = (output_np * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)


def run_esrgan_super_resolution(
    degraded_lr_bgr: np.ndarray,
    target_size: tuple[int, int],
    model_dir: Path,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    执行双三次插值与 ESRGAN 超分恢复。

    参数：
    - degraded_lr_bgr: 低分辨率退化图像
    - target_size: 目标输出尺寸，格式为 (width, height)
    """
    bicubic_bgr = cv2.resize(
        degraded_lr_bgr,
        target_size,
        interpolation=cv2.INTER_CUBIC,
    )
    save_image(output_dir / "bicubic_result.png", bicubic_bgr)

    model_path = model_dir / "RealESRGAN_x4plus.pth"
    download_model_if_needed(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ESRGAN 推理设备: {device}")
    model = load_esrgan_model(model_path, device)
    esrgan_bgr = run_esrgan_inference(degraded_lr_bgr, model, device)

    # 若输出尺寸与原图不同，做一次安全检查与调整
    if (esrgan_bgr.shape[1], esrgan_bgr.shape[0]) != target_size:
        esrgan_bgr = cv2.resize(esrgan_bgr, target_size, interpolation=cv2.INTER_CUBIC)

    save_image(output_dir / "esrgan_result.png", esrgan_bgr)
    return bicubic_bgr, esrgan_bgr
