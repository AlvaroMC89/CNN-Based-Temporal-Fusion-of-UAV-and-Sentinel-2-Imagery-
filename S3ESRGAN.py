#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3-ESRGAN  –  Spectral-Spatial Enhanced Super-Resolution GAN
=============================================================

Pure ESRGAN adapted for 5-band images for the
Sentinel-2-to-UAV fusion task.

GENERATOR architecture (S3ESRGANGenerator):
    1. Input Conv     : Conv2d(C → NF)
    2. RRDB Trunk     : NB_RRDB Residual-in-Residual Dense Blocks
    3. Trunk Conv     : Conv2d(NF → NF) + skip from conv_first
    4. Spectral Head  : 2× Conv1×1(NF → NF) with LeakyReLU
                        (captures correlations across the C channels)
    5. Output Conv    : Conv2d(NF → C)
    6. Residual       : output = x + output_conv(features)

DISCRIMINATOR (S3ESRGANDiscriminator):
    PatchGAN with spectral normalization.
    Output: patch map (B, 1, H//8, W//8).

Training in TWO PHASES:
    Phase 1 (PRETRAIN_EPOCHS): generator only, pixel-level L1 loss.
    Phase 2 (ADV_EPOCHS)     : G + D, Relativistic average GAN (RaGAN)
                              + VGG19 perceptual loss (first 3 channels)
                              + spectral-preserving loss
                              + edge-preservation loss.

Data pipeline:
    - Automatic pairing of Sentinel and UAV images by date and resolution.
    - Random-patch dataset (no temporal sequences).
    - Directory-wide inference with SAM/SID/ERGAS evaluation.
"""

import csv
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision import models as tv_models
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_metrics import (
    read_raster,
    compute_band_metrics,
    spectral_angle_mapper,
    spectral_information_divergence,
    ergas,
    format_table,
    save_csv,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SENTINEL_DIR = Path(r"path\to\sentinel")
DRON_DIR     = Path(r"path\to\dron")
TARGET_RESOLUTION = "2m"   # None → auto-detect

SENTINEL_INFER_DIR = Path(r"path\to\sentinel_inferencia")
DRON_INFER_DIR     = Path(r"path\to\dron_inferencia")

# ── Training parameters ───────────────────────────────────────────────────────
SCALE_FACTOR      = 1
LR_PATCH_SIZE     = 32
HR_PATCH_SIZE     = LR_PATCH_SIZE * SCALE_FACTOR
PATCHES_PER_IMAGE = 200
BATCH_SIZE        = 4
PRETRAIN_EPOCHS    = 100
ADV_EPOCHS        = 400
LEARNING_RATE     = 1e-6
MIN_LR            = 1e-6
GRAD_CLIP_NORM    = 1.0
TEST_RATIO        = 0.1
SEED              = 2025
MAX_PATCH_RETRIES = 20
NUM_WORKERS       = 0 if os.name == "nt" else 4
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA_PSNR = 0.02
EARLY_STOP_MIN_DELTA_ERGAS = 0.02

# ── Model parameters ──────────────────────────────────────────────────────────
NUM_CHANNELS     = 5     # input/output bands
NF               = 64    # number of feature maps
NB_RRDB          = 16    # RRDB blocks in the trunk
GROWTH           = 32    # growth channels in each RDB
RESIDUAL_SCALING = 0.2   # internal RRDB residual scaling

# ── Loss function weights (paper) ─────────────────────────────────────────────
LAMBDA_L1    = 1.0
LAMBDA_ADV   = 0.001
LAMBDA_PER   = 0.1
LAMBDA_SPC   = 0.05
LAMBDA_EDGE  = 0.05

# ── Normalization ranges ──────────────────────────────────────────────────────
INDEX_MIN       = -1.0
INDEX_MAX       =  1.0
NORMALIZE_BANDS = False

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP    = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True
EPS_FINITE = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tif_stem(stem: str):
    """Extract (date, resolution) from '20250709_comp_sentinel_2m'."""
    m = re.match(r"^(\d{8})_comp_\w+_(\d+(?:p\d+|(?:\.\d+)?)m)$", stem, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()


def _index_dir(directory: Path):
    idx = {}
    for p in sorted(directory.glob("*.tif")):
        date, res = _parse_tif_stem(p.stem)
        if date and res:
            idx[(date, res)] = p
    return idx


def build_pairs_from_dirs(sentinel_dir: Path, dron_dir: Path, target_res: str = None):
    """
    Scans sentinel_dir and dron_dir, matches by (date, resolution), and
    returns a chronologically ordered list of (sentinel_path, dron_path).
    """
    s_idx = _index_dir(sentinel_dir)
    d_idx = _index_dir(dron_dir)
    common = set(s_idx) & set(d_idx)
    if not common:
        raise FileNotFoundError(
            f"No matching pairs were found in:\n  {sentinel_dir}\n  {dron_dir}"
        )

    res = target_res
    if res is None:
        res = Counter(r for _, r in common).most_common(1)[0][0]
        print(f"[auto] Detected resolution: {res}")

    pairs = sorted(
        [(date, str(s_idx[(date, res)]), str(d_idx[(date, res)]))
         for date, r in common if r == res],
        key=lambda x: x[0],
    )
    if not pairs:
        raise FileNotFoundError(
            f"No pairs were found for resolution '{res}'. "
            f"Available resolutions: {sorted({r for _, r in common})}"
        )

    print(f"[auto] Found {len(pairs)} pairs for resolution '{res}':")
    for date, sp, dp in pairs:
        print(f"       {date}  sentinel={Path(sp).name}  dron={Path(dp).name}")

    return [(sp, dp) for _, sp, dp in pairs]


def build_inference_pairs(sentinel_infer_dir: Path, dron_infer_dir: Path, target_res: str):
    """Detect inference pairs. Returns [(date, s_path, d_path_or_None)]."""
    s_idx = _index_dir(sentinel_infer_dir)
    d_idx = _index_dir(dron_infer_dir) if dron_infer_dir.exists() else {}

    pairs = []
    for (date, res), sp in sorted(s_idx.items(), key=lambda x: x[0][0]):
        if res == target_res:
            pairs.append((date, sp, d_idx.get((date, res))))

    if not pairs:
        raise FileNotFoundError(
            f"No TIFFs were found for resolution '{target_res}' in:\n  {sentinel_infer_dir}"
        )

    print(f"[inference] Found {len(pairs)} dates for resolution '{target_res}':")
    for date, sp, dp in pairs:
        print(f"             {date}  sentinel={sp.name}  "
              f"dron_ref={dp.name if dp else '(no reference)'}")
    return pairs


# ── Automatic pair detection ──────────────────────────────────────────────────
ALL_PAIRS = build_pairs_from_dirs(SENTINEL_DIR, DRON_DIR, target_res=TARGET_RESOLUTION)


def _extract_res(s: str):
    m = re.search(r"(\d+(?:[.,]\d+|p\d+)?m)", s, flags=re.IGNORECASE)
    return m.group(1).lower().replace("p", ".").replace(",", ".") if m else None


RESOLUTION = Counter(
    r for sp, _ in ALL_PAIRS for r in [_extract_res(Path(sp).stem)] if r
).most_common(1)[0][0]

random.seed(SEED)
random.shuffle(ALL_PAIRS)
n_val       = max(1, int(len(ALL_PAIRS) * TEST_RATIO)) if len(ALL_PAIRS) > 1 else 0
VAL_PAIRS   = ALL_PAIRS[:n_val]
TRAIN_PAIRS = ALL_PAIRS[n_val:]
num_total   = len(ALL_PAIRS)

print(f"[config] Total={num_total} | train={len(TRAIN_PAIRS)} | val={len(VAL_PAIRS)} | "
    f"resolution={RESOLUTION}")

OUT_DIR = Path(r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 2\s3esrgan_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_PATH = OUT_DIR / f"s3esrgan_gen_res{RESOLUTION}_dates{num_total}.pth"
DIS_PATH = OUT_DIR / f"s3esrgan_dis_res{RESOLUTION}_dates{num_total}.pth"


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def clip_to_range(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, INDEX_MIN, INDEX_MAX)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def compute_psnr(mse: float, data_range: float = INDEX_MAX - INDEX_MIN) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def resize_tensor(x: torch.Tensor, size_hw: tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    return F.interpolate(x, size=size_hw, mode=mode, align_corners=False)


def apply_random_augment_pair(lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply identical geometric augmentations to LR and HR."""
    k = random.randint(0, 3)
    if k:
        lr = torch.rot90(lr, k, dims=(1, 2))
        hr = torch.rot90(hr, k, dims=(1, 2))
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=(2,))
        hr = torch.flip(hr, dims=(2,))
    if random.random() < 0.5:
        lr = torch.flip(lr, dims=(1,))
        hr = torch.flip(hr, dims=(1,))
    return lr.contiguous(), hr.contiguous()


def compute_batch_ergas(pred: torch.Tensor, target: torch.Tensor, scale_ratio: float = SCALE_FACTOR) -> float:
    """Approximate batch ERGAS for early validation."""
    pred_np = pred.detach().float().cpu().numpy()
    tgt_np = target.detach().float().cpu().numpy()
    eps = 1e-8
    rmse = np.sqrt(np.mean((pred_np - tgt_np) ** 2, axis=(0, 2, 3)))
    mean_ref = np.mean(np.abs(tgt_np), axis=(0, 2, 3)) + eps
    return float((100.0 / max(scale_ratio, eps)) * np.sqrt(np.mean((rmse / mean_ref) ** 2)))


def spectral_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Penalizes pixel-wise spectral differences using cosine similarity across spectral bands.
    Both tensors have dimensions: (B, C, H, W).
    """
    cos = F.cosine_similarity(pred, target, dim=1, eps=1e-8)
    return (1.0 - cos).mean()


def edge_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Loss de bordes con magnitud Sobel (agregado sobre canales).
    """
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)

    def _edge_mag(x: torch.Tensor) -> torch.Tensor:
        xg = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(xg, sobel_x, padding=1)
        gy = F.conv2d(xg, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    return F.l1_loss(_edge_mag(pred), _edge_mag(target))


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB).
    Progressive feature concatenation + scaled residual connection.
    """

    def __init__(self, nf: int = NF, growth: int = GROWTH, res_scale: float = RESIDUAL_SCALING):
        super().__init__()
        self.c1 = nn.Conv2d(nf,             growth, 3, 1, 1)
        self.c2 = nn.Conv2d(nf +   growth,  growth, 3, 1, 1)
        self.c3 = nn.Conv2d(nf + 2*growth,  growth, 3, 1, 1)
        self.c4 = nn.Conv2d(nf + 3*growth,  nf,     3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res_scale = res_scale
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x,  x1],          dim=1)))
        x3 = self.act(self.c3(torch.cat([x,  x1, x2],      dim=1)))
        x4 =          self.c4(torch.cat([x,  x1, x2, x3],  dim=1))
        return x + self.res_scale * x4


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 RDB en cascada + residual externo."""

    def __init__(self, nf: int = NF, growth: int = GROWTH, res_scale: float = RESIDUAL_SCALING):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, growth, res_scale)
        self.rdb2 = ResidualDenseBlock(nf, growth, res_scale)
        self.rdb3 = ResidualDenseBlock(nf, growth, res_scale)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.rdb3(self.rdb2(self.rdb1(x)))


class ScaleAdaptiveBlock(nn.Module):
    """
    Multi-scale block with dilated branches to capture spatial context.
    """

    def __init__(self, nf: int = NF):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, dilation=1), nn.LeakyReLU(0.2, inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 2, dilation=2), nn.LeakyReLU(0.2, inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 3, dilation=3), nn.LeakyReLU(0.2, inplace=True))
        self.fuse = nn.Conv2d(3 * nf, nf, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.fuse(y)


class SpatialAttention(nn.Module):
    """
    CBAM-style spatial attention to emphasize relevant structures.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class S3ESRGANGenerator(nn.Module):
    """
    S3-ESRGAN generator.

    x(B,C,H,W) ──► conv_first(C→NF) ──► RRDB Trunk ──► trunk_conv
                ──► skip + ──► scale_adaptive ──► spatial_attention
                ──► spectral_head (2×Conv1×1)
                ──► upsample ×5 ──► output_conv(NF→C)
    output = bicubic(x, ×5) + output_conv(features_up)
    """

    def __init__(
        self,
        num_channels: int = NUM_CHANNELS,
        nf: int           = NF,
        nb_rrdb: int      = NB_RRDB,
        growth: int       = GROWTH,
        res_scale: float  = RESIDUAL_SCALING,
    ):
        super().__init__()
        self.num_channels = num_channels

        self.conv_first = nn.Conv2d(num_channels, nf, 3, 1, 1)

        self.trunk      = nn.Sequential(*[RRDB(nf, growth, res_scale) for _ in range(nb_rrdb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        self.scale_adaptive = ScaleAdaptiveBlock(nf)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Spectral head: Conv1×1 models correlations across spectral bands
        self.spectral_head = nn.Sequential(
            nn.Conv2d(nf, nf, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up_refine = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, num_channels, 3, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)  →  (B, C, H*5, W*5)"""
        feat  = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat  = feat + trunk                  # internal skip connection
        feat  = self.scale_adaptive(feat)     # scale-adaptive component
        feat  = self.spatial_attention(feat)  # spatial-attention component
        feat  = self.spectral_head(feat)      # spectral correlations
        feat_up = F.interpolate(feat, scale_factor=SCALE_FACTOR, mode="bilinear", align_corners=False)
        feat_up = self.up_refine(feat_up)
        base = F.interpolate(x, scale_factor=SCALE_FACTOR, mode="bicubic", align_corners=False)
        return base + self.output_conv(feat_up)


class S3ESRGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalization.
    Input: (B, C, H, W) → output: (B, 1, H//8, W//8).
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, nf: int = NF):
        super().__init__()

        def csn(ic, oc, stride=1):
            return nn.utils.spectral_norm(nn.Conv2d(ic, oc, 3, stride, 1, bias=False))

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1),    # first layer without spectral_norm
            nn.LeakyReLU(0.2, inplace=True),

            csn(nf,     nf,     stride=2), nn.LeakyReLU(0.2, inplace=True),
            csn(nf,     nf * 2, stride=1), nn.LeakyReLU(0.2, inplace=True),
            csn(nf * 2, nf * 2, stride=2), nn.LeakyReLU(0.2, inplace=True),
            csn(nf * 2, nf * 4, stride=1), nn.LeakyReLU(0.2, inplace=True),
            csn(nf * 4, nf * 4, stride=2), nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(nf * 4, 1, 3, 1, 1)),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 features through relu5_4 for the perceptual loss.
    Uses the first 3 channels of the multiband image.
    """

    def __init__(self):
        super().__init__()
        if not HAS_TORCHVISION:
            raise RuntimeError("torchvision is not available.")
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT).features
        self.feature = nn.Sequential(*list(vgg.children())[:35]).eval().to(DEVICE)
        for p in self.feature.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SentinelDronDataset(Dataset):
    """
        Dataset of Sentinel–UAV LR-HR patches.
        Each item returns:
            lr_patch: (C, LR_PATCH_SIZE, LR_PATCH_SIZE)
            hr_patch: (C, HR_PATCH_SIZE, HR_PATCH_SIZE)
    """

    def __init__(
        self,
        pairs,
        lr_patch_size: int      = LR_PATCH_SIZE,
        hr_patch_size: int      = HR_PATCH_SIZE,
        num_patches_per_image: int = PATCHES_PER_IMAGE,
        normalize: bool         = NORMALIZE_BANDS,
        max_retry: int          = MAX_PATCH_RETRIES,
        augment: bool           = True,
    ):
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = hr_patch_size
        self.max_retry  = max_retry
        self.augment = augment

        self.data: list[tuple[np.ndarray, np.ndarray]] = []
        for s_path, d_path in pairs:
            with rasterio.open(s_path) as ss, rasterio.open(d_path) as ds:
                s = ss.read().astype(np.float32)
                d = ds.read().astype(np.float32)
                if ss.nodata is not None:
                    s = np.where(s == ss.nodata, np.nan, s)
                if ds.nodata is not None:
                    d = np.where(d == ds.nodata, np.nan, d)
                s = clip_to_range(s)
                d = clip_to_range(d)
                if normalize:
                    for arr in (s, d):
                        for b in range(arr.shape[0]):
                            v, mask = arr[b], np.isfinite(arr[b])
                            if np.any(mask):
                                arr[b] = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
                self.data.append((s, d))

        if not self.data:
            raise ValueError("No data pairs were loaded.")

        self.num_patches = num_patches_per_image * len(self.data)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int):
        hps = self.hr_patch_size
        lps = self.lr_patch_size
        for _ in range(self.max_retry):
            pidx = random.randint(0, len(self.data) - 1)
            s, d = self.data[pidx]
            H, W = s.shape[1], s.shape[2]
            max_r = max(0, H - hps)
            max_c = max(0, W - hps)
            r0 = random.randint(0, max_r)
            c0 = random.randint(0, max_c)

            # In this adaptation, extract an HR patch from both sources and
            # downsample Sentinel to LR for explicit SR×5 training.
            sp_hr = s[:, r0:r0+hps, c0:c0+hps]
            dp = d[:, r0:r0+hps, c0:c0+hps]
            if not np.isfinite(sp_hr).any() or not np.isfinite(dp).any():
                continue

            sp_hr_t = torch.from_numpy(np.nan_to_num(sp_hr, nan=0.0)).float()
            dp_t = torch.from_numpy(np.nan_to_num(dp, nan=0.0)).float()

            sp_lr_t = resize_tensor(sp_hr_t.unsqueeze(0), (lps, lps)).squeeze(0)
            if self.augment:
                sp_lr_t, dp_t = apply_random_augment_pair(sp_lr_t, dp_t)
            return (
                sp_lr_t,
                dp_t,
            )

        # fallback: parche central
        pidx = random.randint(0, len(self.data) - 1)
        s, d = self.data[pidx]
        H, W = s.shape[1], s.shape[2]
        r0   = max(0, (H - hps) // 2)
        c0   = max(0, (W - hps) // 2)
        sp_hr_t = torch.from_numpy(np.nan_to_num(s[:, r0:r0+hps, c0:c0+hps], nan=0.0)).float()
        dp_t = torch.from_numpy(np.nan_to_num(d[:, r0:r0+hps, c0:c0+hps], nan=0.0)).float()
        sp_lr_t = resize_tensor(sp_hr_t.unsqueeze(0), (lps, lps)).squeeze(0)
        return (
            sp_lr_t,
            dp_t,
        )


def create_dataloader(dataset, shuffle=True, drop_last=True):
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = 2
    return DataLoader(dataset, **kwargs)


def _build_loaders():
    full_ds = SentinelDronDataset(
        TRAIN_PAIRS,
        lr_patch_size=LR_PATCH_SIZE,
        hr_patch_size=HR_PATCH_SIZE,
        normalize=NORMALIZE_BANDS,
        augment=True,
    )

    if VAL_PAIRS:
        train_loader = create_dataloader(full_ds, shuffle=True, drop_last=True)
        val_ds = SentinelDronDataset(
            VAL_PAIRS,
            lr_patch_size=LR_PATCH_SIZE,
            hr_patch_size=HR_PATCH_SIZE,
            num_patches_per_image=PATCHES_PER_IMAGE // 2,
            normalize=NORMALIZE_BANDS,
            augment=False,
        )
        val_loader = create_dataloader(val_ds, shuffle=False, drop_last=False)
    else:
        total = len(full_ds)
        n_val = max(1, int(total * TEST_RATIO)) if total > 1 else 0
        n_tr  = total - n_val
        if n_val > 0 and n_tr > 0:
            tr_sub, vl_sub = torch.utils.data.random_split(
                full_ds, [n_tr, n_val],
                generator=torch.Generator().manual_seed(SEED),
            )
            train_loader = create_dataloader(tr_sub, shuffle=True,  drop_last=True)
            val_loader   = create_dataloader(vl_sub, shuffle=False, drop_last=False)
        else:
            train_loader = create_dataloader(full_ds, shuffle=True, drop_last=True)
            val_loader   = None

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_s3esrgan():
    """
        Train S3-ESRGAN in two phases:
            Phase 1 – Generator pretraining with L1.
            Phase 2 – Adversarial training: RaGAN + perceptual + spectral + edge + L1.
    """
    set_seed(SEED)
    if not TRAIN_PAIRS:
        raise ValueError("No training pairs are available.")

    train_loader, val_loader = _build_loaders()

    # ── Models ────────────────────────────────────────────────────────────────
    gen = S3ESRGANGenerator(num_channels=NUM_CHANNELS, nf=NF, nb_rrdb=NB_RRDB).to(DEVICE)
    dis = S3ESRGANDiscriminator(in_channels=NUM_CHANNELS, nf=NF).to(DEVICE)

    n_g = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    n_d = sum(p.numel() for p in dis.parameters() if p.requires_grad)
    print(f"S3-ESRGAN | device={DEVICE} | G={n_g:,} params | D={n_d:,} params")
    print(f"  train_pairs={len(TRAIN_PAIRS)} | val_pairs={len(VAL_PAIRS)}")
    print(f"  dataloader num_workers={NUM_WORKERS} (Windows-safe)")

    # ── Losses ────────────────────────────────────────────────────────────────
    l1_loss = nn.L1Loss()
    bce_logits = nn.BCEWithLogitsLoss()

    feat_extractor = None
    if HAS_TORCHVISION and NUM_CHANNELS >= 3:
        try:
            feat_extractor = VGGFeatureExtractor()
            print("  Perceptual loss (VGG19): enabled")
        except Exception as exc:
            print(f"  Perceptual loss unavailable: {exc}")

    # ── Optimizadores ─────────────────────────────────────────────────────────
    opt_g = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    opt_d = optim.Adam(dis.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    sched_g = optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=max(1, PRETRAIN_EPOCHS), eta_min=MIN_LR
    )
    sched_d = optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=max(1, ADV_EPOCHS), eta_min=MIN_LR
    )

    try:
        scaler_g = torch.amp.GradScaler(device_type="cuda", enabled=USE_AMP)
        scaler_d = torch.amp.GradScaler(device_type="cuda", enabled=USE_AMP)
    except Exception:
        scaler_g = torch.amp.GradScaler(enabled=USE_AMP)
        scaler_d = torch.amp.GradScaler(enabled=USE_AMP)

    best_g_loss = float("inf")
    hist_g, hist_d = [], []

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: Pretraining (L1)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1 – Pretraining ({PRETRAIN_EPOCHS} epochs)")
    print(f"{'='*60}")
    _t0_fase1 = time.time()

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        gen.train()
        ep_loss, nb = 0.0, 0
        bad_batches = 0

        for s_batch, d_batch in train_loader:
            s = s_batch.to(DEVICE, non_blocking=PIN_MEMORY)
            d = d_batch.to(DEVICE, non_blocking=PIN_MEMORY)

            if not (_is_finite_tensor(s) and _is_finite_tensor(d)):
                bad_batches += 1
                continue

            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                fake = gen(s)
                loss = l1_loss(fake, d)

            if not torch.isfinite(loss):
                bad_batches += 1
                continue

            scaler_g.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler_g.unscale_(opt_g)
                nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP_NORM)
            scaler_g.step(opt_g)
            scaler_g.update()

            ep_loss += loss.item()
            nb      += 1

        if nb == 0:
            raise RuntimeError(
                "All pretraining batches were non-finite. "
                "Reduce LEARNING_RATE or disable AMP."
            )
        avg = ep_loss / max(1, nb)
        sched_g.step()

        if epoch % 1 == 0 or epoch == PRETRAIN_EPOCHS:
            print(f"  [Pretrain {epoch:>4}/{PRETRAIN_EPOCHS}] "
                f"L1: {avg:.6f} | PSNR: {compute_psnr(avg):.2f} dB | "
                  f"LR: {opt_g.param_groups[0]['lr']:.2e} | bad_batches: {bad_batches}")
            torch.save(gen.state_dict(), GEN_PATH)

    _t1_fase1 = time.time()
    print(f"Phase 1 completed.  Time: {_t1_fase1 - _t0_fase1:.1f} s  ({(_t1_fase1 - _t0_fase1)/60:.2f} min)")

    # Reset the generator optimizer for the adversarial phase
    opt_g   = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    sched_g = optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=max(1, ADV_EPOCHS), eta_min=MIN_LR
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: Adversarial training
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2 – Adversarial training ({ADV_EPOCHS} epochs)")
    print(f"{'='*60}")
    _t0_fase2 = time.time()
    best_val_psnr = -float("inf")
    best_val_ergas = float("inf")
    no_improve = 0
    for epoch in range(1, ADV_EPOCHS + 1):
        gen.train()
        dis.train()
        ep_g, ep_d, nb = 0.0, 0.0, 0
        bad_batches = 0

        # Gradually ramp up the adversarial weight (stabilizes early epochs)
        adv_w = LAMBDA_ADV * min(1.0, epoch / max(1, ADV_EPOCHS // 10))

        for s_batch, d_batch in train_loader:
            s = s_batch.to(DEVICE, non_blocking=PIN_MEMORY)
            d = d_batch.to(DEVICE, non_blocking=PIN_MEMORY)

            if not (_is_finite_tensor(s) and _is_finite_tensor(d)):
                bad_batches += 1
                continue

            # ── Generator forward pass ────────────────────────────────────
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                fake = gen(s)

            # ── Discriminator (RaGAN, BCE with logits) ────────────────────
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred_real = dis(d)
                pred_fake = dis(fake.detach())
                rm = pred_real.mean()
                fm = pred_fake.mean()
                loss_d = 0.5 * (
                    bce_logits(pred_real - fm, torch.ones_like(pred_real)) +
                    bce_logits(pred_fake - rm, torch.zeros_like(pred_fake))
                )

            if not torch.isfinite(loss_d):
                bad_batches += 1
                continue
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ── Generator (content + perceptual + adversarial) ────────────
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                pred_fake_g = dis(fake)
                pred_real_g = dis(d)
                rm_g = pred_real_g.mean()
                fm_g = pred_fake_g.mean()
                loss_adv = 0.5 * (
                    bce_logits(pred_fake_g - rm_g, torch.ones_like(pred_fake_g)) +
                    bce_logits(pred_real_g - fm_g, torch.zeros_like(pred_real_g))
                )
                loss_l1 = l1_loss(fake, d)

                loss_perc = torch.tensor(0.0, device=DEVICE)
                if feat_extractor is not None:
                    try:
                        f_f = ((fake[:, :3] + 1.0) * 0.5).clamp(0.0, 1.0)
                        f_r = ((d[:, :3]    + 1.0) * 0.5).clamp(0.0, 1.0)
                        loss_perc = F.l1_loss(feat_extractor(f_f), feat_extractor(f_r))
                    except Exception:
                        pass

                loss_spc = spectral_preservation_loss(fake, d)
                loss_edge = edge_preservation_loss(fake, d)

                loss_g = (LAMBDA_L1   * loss_l1 +
                          LAMBDA_PER  * loss_perc +
                          adv_w       * loss_adv +
                          LAMBDA_SPC  * loss_spc +
                          LAMBDA_EDGE * loss_edge)

            if not torch.isfinite(loss_g):
                bad_batches += 1
                continue

            scaler_g.scale(loss_g).backward()
            if GRAD_CLIP_NORM:
                scaler_g.unscale_(opt_g)
                nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP_NORM)
            scaler_g.step(opt_g)
            scaler_g.update()

            ep_g += loss_g.item()
            ep_d += loss_d.item()
            nb   += 1

        if nb == 0:
            raise RuntimeError(
                "All adversarial batches were non-finite. "
                "Reduce LEARNING_RATE or disable AMP."
            )
        avg_g = ep_g / nb
        avg_d = ep_d / nb
        sched_g.step()
        sched_d.step()
        hist_g.append(avg_g)
        hist_d.append(avg_d)

        # ── Per-epoch validation with PSNR and ERGAS ──────────────────────
        val_str = "N/A"
        val_psnr = float("nan")
        val_ergas = float("nan")
        if val_loader is not None:
            gen.eval()
            v_loss, v_psnr, v_ergas, v_n = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for vs, vd in val_loader:
                    vs, vd = vs.to(DEVICE), vd.to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        v_pred = gen(vs)
                        v_loss += l1_loss(v_pred, vd).item()
                        v_mse = F.mse_loss(v_pred, vd).item()
                    v_psnr += compute_psnr(v_mse)
                    v_ergas += compute_batch_ergas(v_pred, vd)
                    v_n += 1
            if v_n > 0:
                val_str = f"{v_loss / v_n:.6f}"
                val_psnr = v_psnr / v_n
                val_ergas = v_ergas / v_n

                improved = False
                if val_psnr > best_val_psnr + EARLY_STOP_MIN_DELTA_PSNR:
                    best_val_psnr = val_psnr
                    improved = True
                if val_ergas < best_val_ergas - EARLY_STOP_MIN_DELTA_ERGAS:
                    best_val_ergas = val_ergas
                    improved = True
                no_improve = 0 if improved else (no_improve + 1)

        if epoch % 1 == 0 or epoch == ADV_EPOCHS:
            print(f"  [Adv {epoch:>4}/{ADV_EPOCHS}] "
                  f"G: {avg_g:.6f} | D: {avg_d:.6f} | "
                  f"Val L1: {val_str} | Val PSNR: {val_psnr:.2f} | Val ERGAS: {val_ergas:.3f} | "
                  f"adv_w: {adv_w:.4f} | bad_batches: {bad_batches} | "
                  f"LR_G: {opt_g.param_groups[0]['lr']:.2e}")

        if avg_g < best_g_loss - 1e-6 or epoch % 1 == 0 or epoch == ADV_EPOCHS:
            best_g_loss = min(best_g_loss, avg_g)
            torch.save(gen.state_dict(), GEN_PATH)
            torch.save(dis.state_dict(), DIS_PATH)

    # ── Final plot ────────────────────────────────────────────────────────────
    try:
        epochs_adv = list(range(1, len(hist_g) + 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs_adv, hist_g, label="G loss",  color="tab:blue")
        ax.plot(epochs_adv, hist_d, label="D loss",  color="tab:orange")
        ax.set_xlabel("Epoch (Phase 2)")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title("S3-ESRGAN – Phase 2 losses")
        plot_path = OUT_DIR / f"training_s3esrgan_{RESOLUTION}_dates{num_total}.png"
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as exc:
        print(f"Could not save the plot: {exc}")

    _t1_fase2 = time.time()
    _total_train = (_t1_fase1 - _t0_fase1) + (_t1_fase2 - _t0_fase2)
    print(f"Phase 2 completed.  Time: {_t1_fase2 - _t0_fase2:.1f} s  ({(_t1_fase2 - _t0_fase2)/60:.2f} min)")
    print("\nS3-ESRGAN training completed.")
    print(f"  Total training time       : {_total_train:.1f} s  ({_total_train/60:.2f} min)")
    print(f"    Phase 1 (pretraining)   : {_t1_fase1 - _t0_fase1:.1f} s")
    print(f"    Phase 2 (adversarial)   : {_t1_fase2 - _t0_fase2:.1f} s")


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(path: str) -> tuple[np.ndarray, dict]:
    """Read, clip, and normalize a TIFF. Returns (arr, profile)."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
    arr = clip_to_range(arr)
    if NORMALIZE_BANDS:
        for b in range(arr.shape[0]):
            v, mask = arr[b], np.isfinite(arr[b])
            arr[b] = ((v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
                      if np.any(mask) else np.zeros_like(v))
    return np.nan_to_num(arr, nan=0.0), profile


def _load_gen(gen_path) -> "S3ESRGANGenerator":
    gen = S3ESRGANGenerator(num_channels=NUM_CHANNELS, nf=NF, nb_rrdb=NB_RRDB).to(DEVICE)
    gen.load_state_dict(torch.load(gen_path, map_location=DEVICE, weights_only=True))
    gen.eval()
    return gen


def _match_reference_shape(pred_arr: np.ndarray, ref_arr: np.ndarray) -> np.ndarray:
    """Resize the reference to the prediction shape for metrics when they differ."""
    if pred_arr.shape[1:] == ref_arr.shape[1:]:
        return ref_arr
    ref_t = torch.from_numpy(ref_arr).unsqueeze(0).float()
    ref_rs = resize_tensor(ref_t, pred_arr.shape[1:]).squeeze(0).numpy().astype(np.float32)
    return ref_rs


@torch.inference_mode()
def apply_s3esrgan_to_full_image(
    sentinel_path,
    out_path,
    gen_path=None,
    evaluate=True,
    csv_out=None,
    reference_dron_path=None,
):
    """
    Apply the S3-ESRGAN generator to a full Sentinel image.

    sentinel_path       : path to the input Sentinel TIFF.
    out_path            : output path for the predicted GeoTIFF.
    reference_dron_path : optional path to the reference UAV TIFF for metrics.
    """
    gen = _load_gen(gen_path or GEN_PATH)
    inp, ref_profile = _preprocess(str(sentinel_path))

    inp_t = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=USE_AMP):
        out = gen(inp_t)

    out_np = clip_to_range(out.squeeze(0).cpu().numpy().astype(np.float32))
    out_np = np.nan_to_num(out_np, nan=-9999.0)

    transform = ref_profile.get("transform")
    if transform is not None and SCALE_FACTOR > 1:
        ref_profile["transform"] = transform * Affine.scale(1.0 / SCALE_FACTOR, 1.0 / SCALE_FACTOR)

    ref_profile.update(
        height=out_np.shape[1],
        width=out_np.shape[2],
        dtype="float32", count=out_np.shape[0], nodata=-9999.0,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path.as_posix(), "w", **ref_profile) as dst:
        dst.write(out_np)
        for i in range(out_np.shape[0]):
            dst.set_band_description(i + 1, f"Band_{i+1}")
    print(f"S3-ESRGAN image saved to: {out_path}")

    if evaluate:
        if reference_dron_path is None:
            stem = Path(sentinel_path).stem
            for sp, dp in ALL_PAIRS:
                if Path(sp).stem == stem:
                    reference_dron_path = dp
                    break

        if reference_dron_path is not None:
            try:
                pred_arr, pred_desc = read_raster(out_path)
                ref_arr,  ref_desc  = read_raster(Path(reference_dron_path))
                ref_arr = _match_reference_shape(pred_arr, ref_arr)
                stats, ref_means = compute_band_metrics(
                    pred_arr, ref_arr, data_range=INDEX_MAX - INDEX_MIN
                )
                descriptions = pred_desc if any(pred_desc) else ref_desc
                print(format_table(stats, descriptions))
                sam_mean, sam_med = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean          = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val         = ergas(stats, ref_means, scale_ratio=1.0)
                print(f"SAM  (mean/median) [°]: {sam_mean:.3f} / {sam_med:.3f}"
                      if np.isfinite(sam_mean) else "SAM: no valid data")
                print(f"SID  (mean): {sid_mean:.6f}"
                      if np.isfinite(sid_mean) else "SID: no valid data")
                print(f"ERGAS: {ergas_val:.3f}"
                      if np.isfinite(ergas_val) else "ERGAS: no valid data")
                if csv_out:
                    save_csv(stats, Path(csv_out), descriptions)
            except Exception as exc:
                print(f"Evaluation failed: {exc}")
        else:
            print("No reference found. Use reference_dron_path to provide one.")


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_on_dir(
    sentinel_infer_dir: Path = SENTINEL_INFER_DIR,
    dron_infer_dir: Path     = DRON_INFER_DIR,
    gen_path=None,
    out_dir: Path = None,
):
    """
    Apply S3-ESRGAN to all dates in sentinel_infer_dir and evaluate
    against the references in dron_infer_dir.
    """
    if out_dir is None:
        out_dir = OUT_DIR / "inferencia"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_pairs = build_inference_pairs(sentinel_infer_dir, dron_infer_dir, RESOLUTION)
    gen = _load_gen(gen_path or GEN_PATH)
    print(f"[inference] Generator loaded from: {gen_path or GEN_PATH}")

    all_metrics = []

    for date, s_path, d_ref_path in infer_pairs:
        print(f"\n{'='*60}")
        print(f"[inference] Date: {date}  →  {s_path.name}")

        inp, ref_profile = _preprocess(str(s_path))
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = gen(inp_t)

        out_np = clip_to_range(out.squeeze(0).cpu().numpy().astype(np.float32))
        out_np = np.nan_to_num(out_np, nan=-9999.0)

        out_tif = out_dir / f"{date}_s3esrgan_{RESOLUTION}.tif"
        csv_out = out_dir / f"{date}_s3esrgan_metrics.csv"

        transform = ref_profile.get("transform")
        if transform is not None and SCALE_FACTOR > 1:
            ref_profile["transform"] = transform * Affine.scale(1.0 / SCALE_FACTOR, 1.0 / SCALE_FACTOR)

        ref_profile.update(
            height=out_np.shape[1],
            width=out_np.shape[2],
            dtype="float32", count=out_np.shape[0], nodata=-9999.0,
            compress="deflate", tiled=True, blockxsize=512, blockysize=512,
        )
        with rasterio.open(out_tif.as_posix(), "w", **ref_profile) as dst:
            dst.write(out_np)
            for i in range(out_np.shape[0]):
                dst.set_band_description(i + 1, f"Band_{i+1}")
        print(f"  → Saved: {out_tif}")

        if d_ref_path is not None:
            try:
                pred_arr, pred_desc = read_raster(out_tif)
                ref_arr,  ref_desc  = read_raster(d_ref_path)
                ref_arr = _match_reference_shape(pred_arr, ref_arr)
                stats, ref_means = compute_band_metrics(
                    pred_arr, ref_arr, data_range=INDEX_MAX - INDEX_MIN
                )
                descriptions = pred_desc if any(pred_desc) else ref_desc
                print(format_table(stats, descriptions))
                sam_mean, sam_med = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean          = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val         = ergas(stats, ref_means, scale_ratio=1.0)
                print(f"  SAM: {sam_mean:.3f}° | SID: {sid_mean:.6f} | ERGAS: {ergas_val:.3f}")
                save_csv(stats, csv_out, descriptions)
                all_metrics.append({"date": date, "SAM": sam_mean,
                                    "SID": sid_mean, "ERGAS": ergas_val})
            except Exception as exc:
                print(f"  [!] Evaluation error for {date}: {exc}")
        else:
            print(f"  No UAV reference for {date}.")

    if all_metrics:
        summary_csv = out_dir / f"resumen_s3esrgan_{RESOLUTION}.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "SAM", "SID", "ERGAS"])
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[inference] Global summary: {summary_csv}")

    print("\n[inference] Completed.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        torch.multiprocessing.freeze_support()
    except Exception:
        pass

    _t0_train = time.time()
    train_s3esrgan()
    _t1_train = time.time()
    print(f"\n[TIMER] Total training time (wall-clock): {_t1_train - _t0_train:.1f} s  "
          f"({(_t1_train - _t0_train)/60:.2f} min)")

    _t0_infer = time.time()
    run_inference_on_dir(
        sentinel_infer_dir=SENTINEL_INFER_DIR,
        dron_infer_dir=DRON_INFER_DIR,
        gen_path=GEN_PATH,
    )
    _t1_infer = time.time()
    print(f"\n[TIMER] Total inference time (wall-clock)  : {_t1_infer - _t0_infer:.1f} s  "
          f"({(_t1_infer - _t0_infer)/60:.2f} min)")
