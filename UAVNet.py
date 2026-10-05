#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-Net (paper-like) for Sentinel-to-UAV spatiotemporal fusion.

Referencia base:
Xiao et al., "Deep Learning-Based Spatiotemporal Fusion of Unmanned Aerial
Vehicle and Satellite Reflectance Images for Crop Monitoring",
IEEE Access, 2023. DOI: 10.1109/ACCESS.2023.3297513.

This implementation reproduces the core architecture described in the paper:
- MResNet encoder (without an initial max-pooling layer)
- Feature Pyramid Network (FPN) without BatchNorm
- Decoder with two ConvTranspose2d layers (k=2, s=2) + 1x1 output layer
- Combined loss: SSIM + L1
- Scheduler WarmupPolyLR

Adaptations required for this project:
- 5 bands (instead of 4)
- Paper-style inputs using [PIt1, PIt2, UIt1]
- Repository data/inference pipeline for date-based Sentinel-UAV pairs
"""

import csv
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

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


# ---------------- SETTINGS ----------------
SENTINEL_DIR = Path(r"path\to\sentinel")
DRON_DIR = Path(r"path\to\dron")
TARGET_RESOLUTION = "2m"
GROUP_BY = "all"

SENTINEL_INFER_DIR = Path(r"path\to\sentinel_inferencia")
DRON_INFER_DIR = Path(r"path\to\dron_inferencia")

SEQ_LENGTH = 4
TEST_RATIO = 0.2
SEED = 2025

PATCHES_PER_SEQ = 200
GRAD_CLIP_NORM = 0.5
CHECKPOINT_INTERVAL = 5
MAX_PATCH_RETRIES = 20
NUM_WORKERS = 4

INDEX_MIN = -1.0
INDEX_MAX = 1.0
NORMALIZE_BANDS = False

NUM_CHANNELS = 5
ENC_OUT_CHANNELS = 256

# Paper-like hyperparameters (DOI: 10.1109/ACCESS.2023.3297513)
# The paper uses patch size 512, batch size 32, initial lr 1e-3, 300 epochs,
# combined SSIM+L1 loss, and the WarmupPolyLR scheduler.
PATCH_SIZE = 96
BATCH_SIZE = 8
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 3
POLY_POWER = 0.9

OUT_DIR = Path(r"path\to\uavnet_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True


# ---------------- DATA UTILITIES ----------------
def _parse_tif_stem(stem: str):
    m = re.match(r"^(\d{8})_comp_\w+_(\d+(?:p\d+|(?:\.\d+)?)m)$", stem, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()


def build_sequences_from_dirs(sentinel_dir: Path, dron_dir: Path, target_res: str = None, group_by: str = "all"):
    def index_dir(directory: Path):
        idx = {}
        for p in sorted(directory.glob("*.tif")):
            date, res = _parse_tif_stem(p.stem)
            if date and res:
                idx[(date, res)] = p
        return idx

    s_idx = index_dir(sentinel_dir)
    d_idx = index_dir(dron_dir)
    common_keys = set(s_idx) & set(d_idx)

    if not common_keys:
        raise FileNotFoundError(
            f"No matching files were found in:\n"
            f"  sentinel: {sentinel_dir}\n  dron: {dron_dir}\n"
            "Check that filenames follow the pattern YYYYMMDD_comp_<sensor>_<res>.tif"
        )

    res_to_use = target_res
    if res_to_use is None:
        res_to_use = Counter(res for _date, res in common_keys).most_common(1)[0][0]
        print(f"[auto] Resolution selected automatically: {res_to_use}")

    pairs = sorted(
        [
            (date, s_idx[(date, res_to_use)], d_idx[(date, res_to_use)])
            for date, res in common_keys
            if res == res_to_use
        ],
        key=lambda x: x[0],
    )

    if not pairs:
        raise FileNotFoundError(
            f"No pairs were found for resolution '{res_to_use}'. "
            f"Available resolutions: {sorted({r for _, r in common_keys})}"
        )

    print(f"[auto] {len(pairs)} pairs found for resolution '{res_to_use}':")
    for date, sp, dp in pairs:
        print(f"       {date}  sentinel={sp.name}  dron={dp.name}")

    if group_by == "year":
        by_year = {}
        for date, sp, dp in pairs:
            by_year.setdefault(date[:4], []).append((str(sp), str(dp)))
        sequences = [seq for seq in by_year.values() if seq]
    else:
        sequences = [[(str(sp), str(dp)) for _, sp, dp in pairs]]

    return sequences


def build_inference_pairs_from_dir(sentinel_infer_dir: Path, dron_infer_dir: Path, target_res: str):
    def index_dir(directory: Path):
        idx = {}
        for p in sorted(directory.glob("*.tif")):
            date, res = _parse_tif_stem(p.stem)
            if date and res:
                idx[(date, res)] = p
        return idx

    s_idx = index_dir(sentinel_infer_dir)
    d_idx = index_dir(dron_infer_dir) if dron_infer_dir.exists() else {}

    pairs = []
    for (date, res), sp in sorted(s_idx.items(), key=lambda x: x[0][0]):
        if res != target_res:
            continue
        dp = d_idx.get((date, res), None)
        pairs.append((date, sp, dp))

    if not pairs:
        raise FileNotFoundError(
            f"No TIFFs were found for resolution '{target_res}' in:\n"
            f"  {sentinel_infer_dir}"
        )

    print(f"[inference] {len(pairs)} dates found for resolution '{target_res}':")
    for date, sp, dp in pairs:
        ref_str = dp.name if dp else "(no reference)"
        print(f"             {date}  sentinel={sp.name}  dron_ref={ref_str}")

    return pairs


def extract_resolution_from_string(s: str):
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+|p\d+)?m)", s, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower().replace("p", ".").replace(",", ".")


def get_resolution_from_pairs(pairs):
    candidates = []
    for s_path, d_path in pairs:
        for p in (s_path, d_path):
            stem = Path(p).stem
            r = extract_resolution_from_string(stem)
            if r:
                candidates.append(r)
            else:
                r = extract_resolution_from_string(str(p))
                if r:
                    candidates.append(r)
    if not candidates:
        return "unknown"
    return Counter(candidates).most_common(1)[0][0]


def clip_to_index_range(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, INDEX_MIN, INDEX_MAX)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ---------------- SPLIT ----------------
ALL_SEQUENCES = build_sequences_from_dirs(
    SENTINEL_DIR,
    DRON_DIR,
    target_res=TARGET_RESOLUTION,
    group_by=GROUP_BY,
)

ALL_PAIRS = [pair for seq in ALL_SEQUENCES for pair in seq]
num_total = len(ALL_PAIRS)
RESOLUTION = get_resolution_from_pairs(ALL_PAIRS)

_train_dates = sorted({Path(s).stem[:8] for s, _ in ALL_PAIRS})
print(f"[config] Training dates ({len(_train_dates)}): {', '.join(_train_dates)}")

random.seed(SEED)
random.shuffle(ALL_SEQUENCES)
num_total_seqs = len(ALL_SEQUENCES)
num_val_seqs = max(1, int(num_total_seqs * TEST_RATIO)) if num_total_seqs > 1 else 0
VAL_SEQUENCES = ALL_SEQUENCES[:num_val_seqs]
TRAIN_SEQUENCES = ALL_SEQUENCES[num_val_seqs:]

VAL_PAIRS = [pair for seq in VAL_SEQUENCES for pair in seq]
TRAIN_PAIRS = [pair for seq in TRAIN_SEQUENCES for pair in seq]

MODEL_FILENAME = f"uavnet_res{RESOLUTION}_dates{num_total}.pth"
MODEL_PATH = OUT_DIR / MODEL_FILENAME

print(
    f"[config] total_sequences={len(ALL_SEQUENCES)} | "
    f"train={len(TRAIN_SEQUENCES)} | val={len(VAL_SEQUENCES)} | "
    f"resolution={RESOLUTION}"
)


# ---------------- DATASET ----------------
class TemporalSentinelDronDataset(Dataset):
    def __init__(
        self,
        sequences,
        patch_size: int = 96,
        num_patches_per_seq: int = 200,
        seq_length=None,
        normalize: bool = False,
        max_invalid_retry: int = MAX_PATCH_RETRIES,
    ):
        self.patch_size = patch_size
        self.seq_length = seq_length
        self.normalize = normalize
        self.max_retry = max_invalid_retry

        self.sequences = []
        for seq in sequences:
            loaded_seq = []
            for s_path, d_path in seq:
                with rasterio.open(s_path) as ss, rasterio.open(d_path) as ds:
                    s_arr = ss.read().astype(np.float32)
                    d_arr = ds.read().astype(np.float32)

                    if s_arr.shape[0] != NUM_CHANNELS:
                        raise ValueError(
                            f"Sentinel image at {s_path} has {s_arr.shape[0]} bands; {NUM_CHANNELS} are required."
                        )
                    if d_arr.shape[0] != NUM_CHANNELS:
                        raise ValueError(
                            f"UAV image at {d_path} has {d_arr.shape[0]} bands; {NUM_CHANNELS} are required."
                        )

                    if ss.nodata is not None:
                        s_arr = np.where(s_arr == ss.nodata, np.nan, s_arr)
                    if ds.nodata is not None:
                        d_arr = np.where(d_arr == ds.nodata, np.nan, d_arr)

                    s_arr = clip_to_index_range(s_arr)
                    d_arr = clip_to_index_range(d_arr)

                    if normalize:
                        for arr in (s_arr, d_arr):
                            for b in range(arr.shape[0]):
                                v = arr[b]
                                mask = np.isfinite(v)
                                if np.any(mask):
                                    mu = np.nanmean(v[mask])
                                    sd = np.nanstd(v[mask]) + 1e-6
                                    arr[b] = (v - mu) / sd

                    loaded_seq.append((s_arr, d_arr))
            self.sequences.append(loaded_seq)

        if not self.sequences:
            raise ValueError("No sequences were loaded.")

        self.H = self.sequences[0][0][0].shape[1]
        self.W = self.sequences[0][0][0].shape[2]

        self.windows = []
        for seq_idx, seq in enumerate(self.sequences):
            T = len(seq)
            win = seq_length if seq_length is not None else T
            if win > T:
                raise ValueError(f"seq_length={win} > sequence length={T}.")
            for start_t in range(T - win + 1):
                self.windows.append((seq_idx, start_t))

        self.num_patches = num_patches_per_seq * len(self.windows)

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        del idx

        ps = self.patch_size
        max_row = self.H - ps
        max_col = self.W - ps
        if max_row <= 0 or max_col <= 0:
            raise ValueError("Image is smaller than PATCH_SIZE.")

        win = self.seq_length

        for _ in range(self.max_retry):
            win_idx = random.randint(0, len(self.windows) - 1)
            seq_idx, start_t = self.windows[win_idx]
            seq = self.sequences[seq_idx]
            T = len(seq) if win is None else win

            r0 = random.randint(0, max_row)
            c0 = random.randint(0, max_col)

            s_patches = []
            d_first = None
            d_last = None
            valid = True

            frames = seq[start_t : start_t + T]
            for t_idx, (s_arr, d_arr) in enumerate(frames):
                sp = s_arr[:, r0 : r0 + ps, c0 : c0 + ps]
                dp = d_arr[:, r0 : r0 + ps, c0 : c0 + ps]

                if (not np.isfinite(sp).any()) or (not np.isfinite(dp).any()):
                    valid = False
                    break

                sp = np.nan_to_num(sp, nan=0.0)
                dp = np.nan_to_num(dp, nan=0.0)
                s_patches.append(sp)
                if t_idx == 0:
                    d_first = dp
                if t_idx == len(frames) - 1:
                    d_last = dp

            if (not valid) or (d_last is None) or (d_first is None):
                continue

            s_seq = torch.from_numpy(np.stack(s_patches, axis=0)).float()
            d_t1 = torch.from_numpy(d_first).float()
            d_t = torch.from_numpy(d_last).float()
            return s_seq, d_t1, d_t

        raise RuntimeError("Could not extract a valid temporal patch.")


# ---------------- UAV-Net MODEL (paper-like) ----------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class MResNet50Encoder(nn.Module):
    """MResNet-50 without an initial max-pooling layer, as described in the paper."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False)

        layers = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPNNoBN(nn.Module):
    """FPN without batch normalization (consistent with the paper)."""

    def __init__(self, c2=256, c3=512, c4=1024, c5=2048, out_channels=ENC_OUT_CHANNELS):
        super().__init__()
        self.lat2 = nn.Conv2d(c2, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, out_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(c5, out_channels, kernel_size=1)

        self.s2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.s3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.s4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.s5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        p5 = self.s5(p5)
        p4 = self.s4(p4)
        p3 = self.s3(p3)
        p2 = self.s2(p2)

        # Use p2 (highest spatial detail) for reconstruction.
        return p2


class UAVDecoder(nn.Module):
    """Decoder with two transposed convolutions (k=2, s=2) and a 1x1 output layer."""

    def __init__(self, in_channels=ENC_OUT_CHANNELS, out_channels=NUM_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class UAVNet(nn.Module):
    """
    UAV-Net paper-like:
    input = [PIt1, PIt2, UIt1], concatenated along the channel dimension.
    """

    def __init__(self, num_channels: int = NUM_CHANNELS):
        super().__init__()
        self.encoder = MResNet50Encoder(in_channels=3 * num_channels)
        self.fpn = FPNNoBN(out_channels=ENC_OUT_CHANNELS)
        self.decoder = UAVDecoder(in_channels=ENC_OUT_CHANNELS, out_channels=num_channels)

    def forward(self, p_t1: torch.Tensor, p_t2: torch.Tensor, u_t1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([p_t1, p_t2, u_t1], dim=1)
        c2, c3, c4, c5 = self.encoder(x)
        fpn_out = self.fpn(c2, c3, c4, c5)
        out = self.decoder(fpn_out)
        out = F.interpolate(out, size=p_t2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.clamp(out, INDEX_MIN, INDEX_MAX)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM loss: 1 - mean SSIM."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / (ssim_d + 1e-8)
    return 1.0 - ssim.mean()


def build_warmup_poly_lr(optimizer, total_epochs: int, warmup_epochs: int = WARMUP_EPOCHS, power: float = POLY_POWER):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, (1.0 - progress) ** power)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------- TRAINING ----------------
def create_dataloader(dataset, shuffle=True, drop_last=True):
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def compute_psnr(mse, data_range=INDEX_MAX - INDEX_MIN):
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def save_training_plots(epochs, train_losses, val_losses, train_psnrs, val_psnrs, out_path: Path):
    e = np.array(epochs, dtype=np.int32)
    t_losses = np.array(train_losses, dtype=np.float32)
    v_losses = np.array(val_losses, dtype=np.float32)
    t_ps = np.array(train_psnrs, dtype=np.float32)
    v_ps = np.array(val_psnrs, dtype=np.float32)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if t_losses.size > 0:
        plt.plot(e, t_losses, label="train_loss")
    if v_losses.size > 0:
        plt.plot(e, v_losses[: e.size], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("UAV-Net - Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    if t_ps.size > 0:
        plt.plot(e, t_ps, label="train_psnr")
    if v_ps.size > 0:
        plt.plot(e, v_ps[: e.size], label="val_psnr")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("UAV-Net - PSNR")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def train_uavnet():
    set_seed(SEED)

    if len(TRAIN_SEQUENCES) == 0:
        raise ValueError("No sequences are available for training.")

    full_train_dataset = TemporalSentinelDronDataset(
        TRAIN_SEQUENCES,
        patch_size=PATCH_SIZE,
        num_patches_per_seq=PATCHES_PER_SEQ,
        seq_length=SEQ_LENGTH,
        normalize=NORMALIZE_BANDS,
        max_invalid_retry=MAX_PATCH_RETRIES,
    )

    train_loader = None
    val_loader = None

    if len(VAL_SEQUENCES) > 0:
        train_loader = create_dataloader(full_train_dataset, shuffle=True, drop_last=True)
        val_dataset = TemporalSentinelDronDataset(
            VAL_SEQUENCES,
            patch_size=PATCH_SIZE,
            num_patches_per_seq=max(1, PATCHES_PER_SEQ // 2),
            seq_length=SEQ_LENGTH,
            normalize=NORMALIZE_BANDS,
            max_invalid_retry=MAX_PATCH_RETRIES,
        )
        val_loader = create_dataloader(val_dataset, shuffle=False, drop_last=False)
    else:
        total_patches = len(full_train_dataset)
        val_count = max(1, int(total_patches * TEST_RATIO)) if total_patches > 1 else 0
        train_count = total_patches - val_count
        if val_count > 0 and train_count > 0:
            train_sub, val_sub = torch.utils.data.random_split(
                full_train_dataset,
                [train_count, val_count],
                generator=torch.Generator().manual_seed(SEED),
            )
            train_loader = create_dataloader(train_sub, shuffle=True, drop_last=True)
            val_loader = create_dataloader(val_sub, shuffle=False, drop_last=False)
            print(f"[val-fallback] Patch-level split enabled: train={train_count} | val={val_count}")
        else:
            train_loader = create_dataloader(full_train_dataset, shuffle=True, drop_last=True)
            print("[val-fallback] Could not create a validation set; ValLoss will remain N/A.")

    model = UAVNet(num_channels=NUM_CHANNELS).to(DEVICE)
    l1_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = build_warmup_poly_lr(optimizer, total_epochs=NUM_EPOCHS)
    try:
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=USE_AMP)
    except Exception:
        scaler = torch.amp.GradScaler() if USE_AMP else torch.amp.GradScaler(enabled=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"UAV-Net on {DEVICE} | parameters: {total_params:,} | "
        f"train_sequences={len(TRAIN_SEQUENCES)} | val_sequences={len(VAL_SEQUENCES)}"
    )

    best_loss = float("inf")
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    t0_train = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for s_seq, d_t1, d_last in train_loader:
            s_seq = s_seq.to(DEVICE, non_blocking=PIN_MEMORY)
            d_t1 = d_t1.to(DEVICE, non_blocking=PIN_MEMORY)
            d_last = d_last.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                p_t1 = s_seq[:, 0]
                p_t2 = s_seq[:, -1]
                out = model(p_t1, p_t2, d_t1)
                loss = ssim_loss(out, d_last) + l1_criterion(out, d_last)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        train_losses.append(avg_loss)
        try:
            train_psnrs.append(compute_psnr(avg_loss))
        except Exception:
            train_psnrs.append(float("nan"))

        val_loss_epoch = float("nan")
        val_psnr_epoch = float("nan")
        if val_loader is not None:
            model.eval()
            v_loss, v_batches, v_mse = 0.0, 0, 0.0
            with torch.no_grad():
                for s_seq, d_t1, d_last in val_loader:
                    s_seq = s_seq.to(DEVICE, non_blocking=PIN_MEMORY)
                    d_t1 = d_t1.to(DEVICE, non_blocking=PIN_MEMORY)
                    d_last = d_last.to(DEVICE, non_blocking=PIN_MEMORY)
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        p_t1 = s_seq[:, 0]
                        p_t2 = s_seq[:, -1]
                        out = model(p_t1, p_t2, d_t1)
                        loss = ssim_loss(out, d_last) + l1_criterion(out, d_last)
                    v_loss += loss.item()
                    v_mse += torch.mean((out - d_last) ** 2).item()
                    v_batches += 1
            if v_batches > 0:
                val_loss_epoch = v_loss / v_batches
                val_psnr_epoch = compute_psnr(v_mse / v_batches)

        val_losses.append(val_loss_epoch)
        val_psnrs.append(val_psnr_epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        val_str = f"{val_loss_epoch:.6f}" if np.isfinite(val_loss_epoch) else "N/A"
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] Loss: {avg_loss:.6f} | Val: {val_str} | LR: {current_lr:.2e}")

        scheduler.step()

        save_model = (avg_loss < best_loss - 1e-6) or (epoch % CHECKPOINT_INTERVAL == 0) or (epoch == NUM_EPOCHS)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
        if save_model:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Model saved: {MODEL_PATH} (best_loss={best_loss:.6f})")

    epochs = list(range(1, len(train_losses) + 1))
    plot_path = OUT_DIR / f"training_metrics_uavnet_res{RESOLUTION}_dates{num_total}.png"
    save_training_plots(epochs, train_losses, val_losses, train_psnrs, val_psnrs, plot_path)
    print(f"Metrics plot saved to: {plot_path}")

    history_csv = OUT_DIR / f"training_history_uavnet_res{RESOLUTION}_dates{num_total}.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_psnr", "val_psnr"])
        for i in range(len(epochs)):
            writer.writerow([epochs[i], train_losses[i], val_losses[i], train_psnrs[i], val_psnrs[i]])
    print(f"Training history saved to: {history_csv}")

    t1_train = time.time()
    print("UAV-Net training finished.")
    print(f"  Total training time: {t1_train - t0_train:.1f} s  ({(t1_train - t0_train)/60:.2f} min)")


# ---------------- INFERENCE ----------------
@torch.inference_mode()
def apply_uavnet_to_full_image(
    sentinel_sequence,
    out_path,
    model_path=None,
    evaluate=True,
    csv_out=None,
    reference_dron_path=None,
    uav_t1_path=None,
):
    model = UAVNet(num_channels=NUM_CHANNELS).to(DEVICE)
    load_path = model_path if model_path is not None else MODEL_PATH
    model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    model.eval()

    arrs = []
    ref_profile = None
    for s_path in sentinel_sequence:
        with rasterio.open(s_path) as src:
            if ref_profile is None:
                ref_profile = src.profile.copy()
            arr = src.read().astype(np.float32)
            if arr.shape[0] != NUM_CHANNELS:
                raise ValueError(
                    f"{Path(s_path).name} has {arr.shape[0]} bands; the model requires {NUM_CHANNELS}."
                )
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            arr = clip_to_index_range(arr)
            if NORMALIZE_BANDS:
                for b in range(arr.shape[0]):
                    v, mask = arr[b], np.isfinite(arr[b])
                    if np.any(mask):
                        arr[b] = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
                    else:
                        arr[b] = np.zeros_like(v)
            arrs.append(np.nan_to_num(arr, nan=0.0))

    s_np = np.stack(arrs, axis=0)
    s_t = torch.from_numpy(s_np).unsqueeze(0).to(DEVICE)
    p_t1 = s_t[:, 0]
    p_t2 = s_t[:, -1]

    if uav_t1_path is None:
        first_sentinel = Path(sentinel_sequence[0]).stem
        for s_p, d_p in ALL_PAIRS:
            if Path(s_p).stem == first_sentinel:
                uav_t1_path = d_p
                break

    if uav_t1_path is None:
        raise ValueError(
            "Could not resolve UIt1. Provide uav_t1_path or include a context date with a known UAV image."
        )

    with rasterio.open(uav_t1_path) as src_u:
        u_arr = src_u.read().astype(np.float32)
        if src_u.nodata is not None:
            u_arr = np.where(u_arr == src_u.nodata, np.nan, u_arr)
        u_arr = clip_to_index_range(u_arr)
        u_arr = np.nan_to_num(u_arr, nan=0.0)
    u_t1 = torch.from_numpy(u_arr).unsqueeze(0).to(DEVICE)

    with torch.amp.autocast("cuda", enabled=USE_AMP):
        out = model(p_t1, p_t2, u_t1)

    out_np = out.detach().squeeze(0).cpu().numpy().astype(np.float32)
    out_np = clip_to_index_range(out_np)
    out_np = np.nan_to_num(out_np, nan=-9999.0)

    ref_profile.update(
        dtype="float32",
        count=out_np.shape[0],
        nodata=-9999.0,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path.as_posix(), "w", **ref_profile) as dst:
        dst.write(out_np)
        for i in range(out_np.shape[0]):
            dst.set_band_description(i + 1, f"Band_{i + 1}")
    print(f"UAV-Net-SR image saved to: {out_path}")

    if evaluate:
        if reference_dron_path is None:
            last_sentinel = Path(sentinel_sequence[-1])
            for s_p, d_p in ALL_PAIRS:
                if Path(s_p).stem == last_sentinel.stem:
                    reference_dron_path = d_p
                    break

        if reference_dron_path is not None:
            try:
                pred_arr, pred_desc = read_raster(out_path)
                ref_arr, ref_desc = read_raster(Path(reference_dron_path))
                stats, ref_means = compute_band_metrics(pred_arr, ref_arr, data_range=INDEX_MAX - INDEX_MIN)
                descriptions = pred_desc if any(pred_desc) else ref_desc
                print(format_table(stats, descriptions))

                sam_mean, sam_med = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val = ergas(stats, ref_means, scale_ratio=1.0)

                print("\n--- UAV-Net global metrics (post-inference) ---")
                print(f"SAM (mean/median) [degrees]: {sam_mean:.3f} / {sam_med:.3f}" if np.isfinite(sam_mean) else "SAM: no valid data")
                print(f"SID (mean): {sid_mean:.6f}" if np.isfinite(sid_mean) else "SID: no valid data")
                print(f"ERGAS: {ergas_val:.3f}" if np.isfinite(ergas_val) else "ERGAS: no valid data")

                if csv_out:
                    save_csv(stats, Path(csv_out), descriptions)
                    print(f"Metrics saved to: {csv_out}")
            except Exception as exc:
                print(f"Could not evaluate the inferred image: {exc}")
        else:
            print("No automatic reference was found. Use reference_dron_path to provide one manually.")


# ---------------- DIRECTORY INFERENCE ----------------
@torch.inference_mode()
def run_inference_on_dir(
    sentinel_infer_dir: Path = SENTINEL_INFER_DIR,
    dron_infer_dir: Path = DRON_INFER_DIR,
    model_path=None,
    out_dir: Path = None,
):
    if out_dir is None:
        out_dir = OUT_DIR / "inferencia"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = RESOLUTION
    infer_pairs = build_inference_pairs_from_dir(sentinel_infer_dir, dron_infer_dir, target_res=res)
    infer_dates = [d for d, _s, _r in infer_pairs]
    print(f"[inference] INFERENCE DATES ({len(infer_dates)}): {', '.join(infer_dates)}")

    train_sentinel_by_date = {Path(s).stem[:8]: s for s, _d in ALL_PAIRS}
    train_sentinel_paths_ordered = [train_sentinel_by_date[d] for d in sorted(train_sentinel_by_date)]

    load_path = model_path if model_path is not None else MODEL_PATH
    model_infer = UAVNet(num_channels=NUM_CHANNELS).to(DEVICE)
    model_infer.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    model_infer.eval()
    print(f"[inference] Model loaded from: {load_path}")

    all_metrics = []
    per_image_times = []
    t0_all = time.time()

    train_dron_by_date = {Path(s).stem[:8]: d for s, d in ALL_PAIRS}

    for date, s_infer_path, d_ref_path in infer_pairs:
        print("\n" + "=" * 60)
        print(f"[inference] Date: {date} -> {s_infer_path.name}")

        max_context = (SEQ_LENGTH - 1) if SEQ_LENGTH is not None else None
        context_all = [p for p in train_sentinel_paths_ordered if Path(p).stem[:8] < date]
        context = context_all[-max_context:] if max_context is not None else context_all
        sequence = context + [str(s_infer_path)]

        if context:
            print(
                f"           Context ({len(context)}): "
                f"{[Path(p).stem[:8] for p in context]} + [{date}] (T={len(sequence)})"
            )
        else:
            print(f"           No previous context -> single-frame mode (T={len(sequence)})")

        arrs = []
        ref_profile = None
        for sp in sequence:
            with rasterio.open(sp) as src:
                if str(sp) == str(s_infer_path):
                    ref_profile = src.profile.copy()
                arr = src.read().astype(np.float32)
                if arr.shape[0] != NUM_CHANNELS:
                    raise ValueError(
                        f"{Path(sp).name} has {arr.shape[0]} bands; the model requires {NUM_CHANNELS}."
                    )
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                arr = clip_to_index_range(arr)
                if NORMALIZE_BANDS:
                    for b in range(arr.shape[0]):
                        v, mask = arr[b], np.isfinite(arr[b])
                        if np.any(mask):
                            arr[b] = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
                        else:
                            arr[b] = np.zeros_like(v)
                arrs.append(np.nan_to_num(arr, nan=0.0))

        # UIt1: most recent UAV image before the inference date.
        prev_dates = [d for d in sorted(train_dron_by_date) if d < date]
        if not prev_dates:
            print("  [!] No previous UIt1 is available; skipping date.")
            continue
        uav_t1_path = train_dron_by_date[prev_dates[-1]]
        with rasterio.open(uav_t1_path) as src_u:
            u_arr = src_u.read().astype(np.float32)
            if src_u.nodata is not None:
                u_arr = np.where(u_arr == src_u.nodata, np.nan, u_arr)
            u_arr = clip_to_index_range(u_arr)
            if NORMALIZE_BANDS:
                for b in range(u_arr.shape[0]):
                    v, mask = u_arr[b], np.isfinite(u_arr[b])
                    if np.any(mask):
                        u_arr[b] = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
                    else:
                        u_arr[b] = np.zeros_like(v)
            u_arr = np.nan_to_num(u_arr, nan=0.0)

        s_np = np.stack(arrs, axis=0)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(DEVICE)
        p_t1 = s_t[:, 0]
        p_t2 = s_t[:, -1]
        u_t1 = torch.from_numpy(u_arr).unsqueeze(0).to(DEVICE)

        t0_img = time.time()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = model_infer(p_t1, p_t2, u_t1)
        t1_img = time.time()

        img_time = t1_img - t0_img
        per_image_times.append((date, img_time))

        out_np = out.detach().squeeze(0).cpu().numpy().astype(np.float32)
        out_np = clip_to_index_range(out_np)
        out_np = np.nan_to_num(out_np, nan=-9999.0)

        out_tif = out_dir / f"{date}_uavnet_{res}.tif"
        csv_out = out_dir / f"{date}_uavnet_metrics.csv"

        ref_profile.update(
            dtype="float32",
            count=out_np.shape[0],
            nodata=-9999.0,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )
        with rasterio.open(out_tif.as_posix(), "w", **ref_profile) as dst:
            dst.write(out_np)
            for i in range(out_np.shape[0]):
                dst.set_band_description(i + 1, f"Band_{i + 1}")

        print(f"  -> Saved: {out_tif}")
        print(f"  -> Image inference time: {img_time:.2f} s")

        if d_ref_path is not None:
            try:
                pred_arr, pred_desc = read_raster(out_tif)
                ref_arr, ref_desc = read_raster(d_ref_path)
                stats, ref_means = compute_band_metrics(pred_arr, ref_arr, data_range=INDEX_MAX - INDEX_MIN)
                descriptions = pred_desc if any(pred_desc) else ref_desc
                print(format_table(stats, descriptions))

                sam_mean, sam_med = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val = ergas(stats, ref_means, scale_ratio=1.0)

                print(
                    f"SAM (mean/median) [degrees]: {sam_mean:.3f} / {sam_med:.3f}"
                    if np.isfinite(sam_mean)
                    else "SAM: no valid data"
                )
                print(
                    f"SID (mean): {sid_mean:.6f}"
                    if np.isfinite(sid_mean)
                    else "SID: no valid data"
                )
                print(
                    f"ERGAS: {ergas_val:.3f}"
                    if np.isfinite(ergas_val)
                    else "ERGAS: no valid data"
                )

                save_csv(stats, csv_out, descriptions)
                print(f"  -> Metrics: {csv_out}")

                all_metrics.append(
                    {
                        "date": date,
                        "SAM": sam_mean,
                        "SID": sid_mean,
                        "ERGAS": ergas_val,
                        "inference_seconds": img_time,
                    }
                )
            except Exception as exc:
                print(f"  [!] Evaluation error for {date}: {exc}")
        else:
            print("  [!] No UAV reference is available; skipping evaluation.")

    t1_all = time.time()
    total_infer_s = t1_all - t0_all

    if all_metrics:
        summary_csv = out_dir / f"resumen_inferencia_uavnet_{res}.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "SAM", "SID", "ERGAS", "inference_seconds"])
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[inference] Global summary saved to: {summary_csv}")

    times_csv = out_dir / f"inference_times_uavnet_{res}.csv"
    with open(times_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "seconds"])
        for date, sec in per_image_times:
            writer.writerow([date, sec])
    print(f"[inference] Per-image times saved to: {times_csv}")

    print("\n[inference] Processing completed.")
    print(f"[TIMER] Total inference time (wall-clock): {total_infer_s:.1f} s ({total_infer_s/60:.2f} min)")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    t0_train = time.time()
    train_uavnet()
    t1_train = time.time()
    print(f"\n[TIMER] Total training time (wall-clock): {t1_train - t0_train:.1f} s ({(t1_train - t0_train)/60:.2f} min)")

    t0_infer = time.time()
    run_inference_on_dir(
        sentinel_infer_dir=SENTINEL_INFER_DIR,
        dron_infer_dir=DRON_INFER_DIR,
        model_path=MODEL_PATH,
    )
    t1_infer = time.time()
    print(f"[TIMER] Total inference time (wall-clock): {t1_infer - t0_infer:.1f} s ({(t1_infer - t0_infer)/60:.2f} min)")
