#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSent – Simultaneous spectral and temporal fusion for Sentinel-to-UAV super-resolution.

Based on:
    Tarasiewicz et al., "Multitemporal and multispectral data fusion for
    super-resolution of Sentinel-2 images", IEEE TGRS 2023, arXiv:2301.11154.

Adaptation for Sentinel-2-to-UAV guided fusion:
    - Input   : temporal sequences of Sentinel-2 composites
                (NDVI, GNDVI, NDRE, NDWI, NDVIre, CIre, ...) already at 2 m.
    - Target  : UAV composite for the last frame (direct supervision).
    - SR scale: 1× (the images already have the same GSD; the model learns
                to transfer radiometry and texture to the UAV domain).

Main architecture (key difference from TVDSR):
    The temporal and spectral branches are processed IN PARALLEL and then fused,
    rather than sequentially (Encoder → LSTM → Decoder).

    Temporal Branch (Conv3D):
        (B, T, C, H, W) → permute → (B, C, T, H, W)
        Conv3D stack (spatiotemporal mixing)
        AdaptiveMaxPool3D(T→1)
        Additional Conv2D
        → (B, FEAT, H, W)

    Spectral Branch (Conv2D over (B·T, C, H, W)):
        (B, T, C, H, W) → reshape → (B·T, C, H, W)
        Conv2D stack (spectral band mixing)
        Reshape back → (B, T, FEAT, H, W)
        MaxPool over T
        Additional Conv2D
        → (B, FEAT, H, W)

    FusionDecoder:
        Concatenate (T_feat, S_feat) → (B, 2·FEAT, H, W)
        1×1 reduction + deep Conv2D layers
        → out (B, C, H, W)
"""

import os
from pathlib import Path
import random
import time
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import math
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


# ============================================================
# CONFIGURATION (identical to TVDSR for comparability)
# ============================================================
SENTINEL_DIR = Path(r"path\to\sentinel")
DRON_DIR     = Path(r"path\to\dron")

TARGET_RESOLUTION = "2m"
GROUP_BY          = "all"

SENTINEL_INFER_DIR = Path(r"path\to\sentinel_inferencia")
DRON_INFER_DIR     = Path(r"path\to\dron_inferencia")

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LENGTH      = 8    # fixed temporal window length (training and inference)
PATCH_SIZE      = 96
PATCHES_PER_SEQ = 200   # patches per window per epoch
BATCH_SIZE      = 8
NUM_EPOCHS      = 500
LEARNING_RATE   = 0.001
GRAD_CLIP_NORM  = 1.0
LR_STEP_SIZE    = 50
LR_GAMMA        = 0.5
CHECKPOINT_INTERVAL = 5
MAX_PATCH_RETRIES   = 20
NUM_WORKERS     = 4
TEST_RATIO      = 0.2
SEED            = 2025

INDEX_MIN       = -1.0
INDEX_MAX       =  1.0
NORMALIZE_BANDS = False

NUM_CHANNELS  = 5   # Sentinel/UAV bands
FEAT_CHANNELS = 64  # feature channels in both branches

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP   = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True


# ============================================================
# UTILITIES (identical to TVDSR)
# ============================================================
def clip_to_index_range(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, INDEX_MIN, INDEX_MAX)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def _autocast():
    """Autocast context compatible with PyTorch < 1.10 and >= 2.0."""
    try:
        return torch.amp.autocast("cuda", enabled=USE_AMP)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=USE_AMP)


def _parse_tif_stem(stem: str):
    import re
    m = re.match(r"^(\d{8})_comp_\w+_(\d+(?:p\d+|(?:\.\d+)?)m)$", stem, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()


def build_sequences_from_dirs(sentinel_dir, dron_dir, target_res=None, group_by="all"):
    def index_dir(directory):
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
            f"No matching files were found.\n"
            f"  sentinel: {sentinel_dir}\n  dron: {dron_dir}"
        )

    res_to_use = target_res
    if res_to_use is None:
        from collections import Counter as _C
        res_to_use = _C(res for _, res in common_keys).most_common(1)[0][0]
        print(f"[auto] Selected resolution: {res_to_use}")

    pairs = sorted(
        [(date, s_idx[(date, res_to_use)], d_idx[(date, res_to_use)])
         for date, res in common_keys if res == res_to_use],
        key=lambda x: x[0],
    )
    if not pairs:
        raise FileNotFoundError(f"No pairs were found for resolution '{res_to_use}'.")

    print(f"[auto] Found {len(pairs)} pairs for resolution '{res_to_use}':")
    for date, sp, dp in pairs:
        print(f"       {date}  sentinel={sp.name}  dron={dp.name}")

    if group_by == "year":
        from collections import defaultdict as _dd
        by_year = _dd(list)
        for date, sp, dp in pairs:
            by_year[date[:4]].append((str(sp), str(dp)))
        return [seq for seq in by_year.values() if seq]
    else:
        return [[(str(sp), str(dp)) for _, sp, dp in pairs]]


def build_inference_pairs_from_dir(sentinel_infer_dir, dron_infer_dir, target_res):
    def index_dir(directory):
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
        pairs.append((date, sp, d_idx.get((date, res), None)))

    if not pairs:
        raise FileNotFoundError(
            f"No TIFFs were found for resolution '{target_res}' in {sentinel_infer_dir}"
        )

    print(f"[inference] Found {len(pairs)} dates for resolution '{target_res}':")
    for date, sp, dp in pairs:
        print(f"             {date}  sentinel={sp.name}  ref={dp.name if dp else '(no reference)'}")
    return pairs


# ── Automatic sequence detection ──────────────────────────────────────────────
TRAIN_SEQUENCES = build_sequences_from_dirs(
    SENTINEL_DIR, DRON_DIR,
    target_res=TARGET_RESOLUTION,
    group_by=GROUP_BY,
)

ALL_SEQUENCES = TRAIN_SEQUENCES.copy()
ALL_PAIRS = [pair for seq in ALL_SEQUENCES for pair in seq]

import re
from collections import Counter

def _get_resolution(pairs):
    candidates = []
    for s_path, d_path in pairs:
        for p in (s_path, d_path):
            m = re.search(r"(\d+(?:[.,]\d+|p\d+)?m)", str(p), flags=re.IGNORECASE)
            if m:
                candidates.append(m.group(1).lower().replace("p", ".").replace(",", "."))
    return Counter(candidates).most_common(1)[0][0] if candidates else "unknown"

RESOLUTION  = _get_resolution(ALL_PAIRS)
num_total   = len(ALL_PAIRS)

_train_dates = sorted({Path(s).stem[:8] for s, _d in ALL_PAIRS})
print(f"[config] TRAINING DATES ({len(_train_dates)}): {', '.join(_train_dates)}")

# Train/validation split by sequence
random.seed(SEED)
random.shuffle(ALL_SEQUENCES)
num_val_seqs    = max(1, int(len(ALL_SEQUENCES) * TEST_RATIO)) if len(ALL_SEQUENCES) > 1 else 0
VAL_SEQUENCES   = ALL_SEQUENCES[:num_val_seqs]
TRAIN_SEQUENCES = ALL_SEQUENCES[num_val_seqs:]
VAL_PAIRS       = [p for seq in VAL_SEQUENCES   for p in seq]
TRAIN_PAIRS     = [p for seq in TRAIN_SEQUENCES for p in seq]

OUT_DIR        = Path(r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 2\deepsent_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = f"deepsent_res{RESOLUTION}_dates{num_total}.pth"
MODEL_PATH     = OUT_DIR / MODEL_FILENAME


# ============================================================
# DeepSent ARCHITECTURE
# ============================================================

class TemporalBranch(nn.Module):
    """
    Temporal Branch: extracts features along the temporal dimension
    T using 3D convolutions.

    - Input   : (B, T, C, H, W)
        - Internally permutes to (B, C, T, H, W) to match the Conv3d
            convention (channels in dimension 1).
        - AdaptiveMaxPool3d collapses T→1, making the branch invariant to T.
        - Output  : (B, FEAT, H, W)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, feat: int = FEAT_CHANNELS):
        super().__init__()
        self.conv3d = nn.Sequential(
            # ── Initial spatial feature extraction (no temporal mixing) ─────
            nn.Conv3d(in_channels, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            # ── Spatiotemporal mixing ─────────────────────────────────
            nn.Conv3d(32, feat, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(feat, feat, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(feat, feat, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
        )
        # Collapse T: supports arbitrary T (including T=1 in single-frame mode)
        self.pool_t = nn.AdaptiveMaxPool3d(output_size=(1, None, None))

        # 2D refinement after temporal collapse
        self.refine = nn.Sequential(
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq : (B, T, C, H, W)  →  (B, FEAT, H, W)"""
        B, T, C, H, W = x_seq.shape
        # (B, T, C, H, W) → (B, C, T, H, W)
        x = x_seq.permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv3d(x)                        # (B, FEAT, T', H, W)
        x = self.pool_t(x)                        # (B, FEAT, 1,  H, W)
        x = x.squeeze(2)                          # (B, FEAT, H, W)
        return self.refine(x)


class SpectralBranch(nn.Module):
    """
    Spectral Branch: extracts correlations between spectral bands using
    Conv2D. It first uses a 1×1 convolution to mix bands without affecting
    spatial dimensions, then 3×3 convolutions to combine spatial and spectral
    information.

    All T frames are processed TOGETHER (reshape to B·T) and aggregated at the end.

    - Input  : (B, T, C, H, W)
    - Output : (B, FEAT, H, W)
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, feat: int = FEAT_CHANNELS):
        super().__init__()
        self.spectral_mix = nn.Sequential(
            # Pure spectral mixing (no spatial change)
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, feat, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        # Refinement after temporal aggregation
        self.refine = nn.Sequential(
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq : (B, T, C, H, W)  →  (B, FEAT, H, W)"""
        B, T, C, H, W = x_seq.shape
        # Flatten the temporal dimension → process all frames as a batch
        x = x_seq.view(B * T, C, H, W)           # (B·T, C, H, W)
        x = self.spectral_mix(x)                  # (B·T, FEAT, H, W)
        x = self.spatial(x)                       # (B·T, FEAT, H, W)
        # Restore the temporal dimension and aggregate (max over T)
        x = x.view(B, T, FEAT_CHANNELS, H, W)    # (B, T, FEAT, H, W)
        x, _ = x.max(dim=1)                       # (B, FEAT, H, W)
        return self.refine(x)


class FusionDecoder(nn.Module):
    """
    Fusion and reconstruction module.

    Receives the feature maps from both branches (temporal + spectral),
    concatenates them, and decodes them into the output band space.

    Input  : (B, 2·FEAT, H, W)
    Output : (B, C, H, W)
    """

    def __init__(
        self,
        in_feat: int    = 2 * FEAT_CHANNELS,
        feat: int       = FEAT_CHANNELS,
        out_channels: int = NUM_CHANNELS,
        num_layers: int  = 10,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_feat, feat, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv2d(feat, feat, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(feat, out_channels, kernel_size=3, padding=1, bias=True))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        """
        t_feat, s_feat : (B, FEAT, H, W)
        Returns output : (B, C, H, W)
        """
        fused = torch.cat([t_feat, s_feat], dim=1)  # (B, 2·FEAT, H, W)
        return self.net(fused)


class DeepSent(nn.Module):
    """
    DeepSent adapted for Sentinel-to-UAV guided fusion.

    Fundamental difference from TVDSR:
        TVDSR  →  Encoder → ConvLSTM (sequential) → Decoder
        DeepSent →  Temporal_Branch ──┐
                                    ├─→ FusionDecoder → out
                    Spectral_Branch ─┘

    The temporal and spectral branches run IN PARALLEL, capturing different
    information sources simultaneously rather than sequentially.
    This prevents spectral information from being "forgotten" while passing
    through the temporal chain, and vice versa.

    Data flow:
        x_seq (B, T, C, H, W)
            ├─→ TemporalBranch  → t_feat (B, F, H, W)
            └─→ SpectralBranch  → s_feat (B, F, H, W)
                    t_feat ++ s_feat → FusionDecoder → out (B, C, H, W)
    """

    def __init__(
        self,
        num_channels: int  = NUM_CHANNELS,
        feat_channels: int = FEAT_CHANNELS,
        decoder_layers: int = 10,
    ):
        super().__init__()
        self.temporal_branch  = TemporalBranch(num_channels, feat_channels)
        self.spectral_branch  = SpectralBranch(num_channels, feat_channels)
        self.fusion_decoder   = FusionDecoder(
            in_feat     = 2 * feat_channels,
            feat        = feat_channels,
            out_channels= num_channels,
            num_layers  = decoder_layers,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq : (B, T, C, H, W)
        Returns prediction: (B, C, H, W)
        """
        t_feat   = self.temporal_branch(x_seq)   # (B, F, H, W)
        s_feat   = self.spectral_branch(x_seq)   # (B, F, H, W)
        out = self.fusion_decoder(t_feat, s_feat)  # (B, C, H, W)
        return out


# ============================================================
# DATASET (identical to TVDSR)
# ============================================================
class TemporalSentinelDronDataset(Dataset):
    """
    Dataset of Sentinel–UAV temporal patches.

    Each item returns:
        s_seq (T, C, H, W)  — sequence of Sentinel patches
        d_last (C, H, W)    — UAV patch for the LAST frame (target)
    """

    def __init__(
        self,
        sequences,
        patch_size: int = 64,
        num_patches_per_seq: int = 200,
        seq_length=None,
        normalize: bool = True,
        max_invalid_retry: int = MAX_PATCH_RETRIES,
    ):
        self.patch_size = patch_size
        self.seq_length = seq_length
        self.normalize  = normalize
        self.max_retry  = max_invalid_retry

        self.sequences = []
        for seq in sequences:
            loaded_seq = []
            for s_path, d_path in seq:
                with rasterio.open(s_path) as ss, rasterio.open(d_path) as ds:
                    s_arr = ss.read().astype(np.float32)
                    d_arr = ds.read().astype(np.float32)
                    if ss.nodata is not None:
                        s_arr = np.where(s_arr == ss.nodata, np.nan, s_arr)
                    if ds.nodata is not None:
                        d_arr = np.where(d_arr == ds.nodata, np.nan, d_arr)
                    s_arr = clip_to_index_range(s_arr)
                    d_arr = clip_to_index_range(d_arr)
                    if normalize:
                        for arr in (s_arr, d_arr):
                            for b in range(arr.shape[0]):
                                v, mask = arr[b], np.isfinite(arr[b])
                                if np.any(mask):
                                    arr[b] = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6)
                    loaded_seq.append((s_arr, d_arr))
            self.sequences.append(loaded_seq)

        if not self.sequences:
            raise ValueError("No sequences were loaded.")

        self.H = self.sequences[0][0][0].shape[1]
        self.W = self.sequences[0][0][0].shape[2]

        self.windows = []
        for seq_idx, seq in enumerate(self.sequences):
            T_seq = len(seq)
            win = seq_length if seq_length is not None else T_seq
            if win > T_seq:
                raise ValueError(f"seq_length={win} > sequence length={T_seq}.")
            for start_t in range(T_seq - win + 1):
                self.windows.append((seq_idx, start_t))

        self.num_patches = num_patches_per_seq * len(self.windows)

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        ps      = self.patch_size
        max_row = self.H - ps
        max_col = self.W - ps
        if max_row <= 0 or max_col <= 0:
            raise ValueError("Image is smaller than PATCH_SIZE.")

        win = self.seq_length

        for _ in range(self.max_retry):
            win_idx          = random.randint(0, len(self.windows) - 1)
            seq_idx, start_t = self.windows[win_idx]
            seq              = self.sequences[seq_idx]
            T_eff            = len(seq) if win is None else win
            r0               = random.randint(0, max_row)
            c0               = random.randint(0, max_col)

            s_patches, d_last, valid = [], None, True
            for t_idx, (s_arr, d_arr) in enumerate(seq[start_t: start_t + T_eff]):
                sp = s_arr[:, r0:r0+ps, c0:c0+ps]
                dp = d_arr[:, r0:r0+ps, c0:c0+ps]
                if not np.isfinite(sp).any() or not np.isfinite(dp).any():
                    valid = False
                    break
                s_patches.append(np.nan_to_num(sp, nan=0.0))
                if t_idx == T_eff - 1:
                    d_last = np.nan_to_num(dp, nan=0.0)

            if not valid or d_last is None:
                continue

            s_seq = torch.from_numpy(np.stack(s_patches, axis=0)).float()
            d_t   = torch.from_numpy(d_last).float()
            return s_seq, d_t

        raise RuntimeError("Could not extract a valid temporal patch.")


# ============================================================
# TRAINING
# ============================================================
def create_dataloader(dataset, shuffle=True, drop_last=True):
    kw = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        pin_memory=PIN_MEMORY,
    )
    if NUM_WORKERS > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(dataset, **kw)


def compute_psnr(mse, data_range=INDEX_MAX - INDEX_MIN):
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def train_deepsent():
    """
    Trains the DeepSent model on temporal sequences of Sentinel–UAV pairs.
    Loss function: MSE between DeepSent(s_seq) and the UAV image for the last frame.
    """
    set_seed(SEED)
    if not TRAIN_SEQUENCES:
        raise ValueError("No sequences are available for training.")

    # ── Dataset ────────────────────────────────────────────────────────────────
    full_ds = TemporalSentinelDronDataset(
        TRAIN_SEQUENCES,
        patch_size=PATCH_SIZE,
        num_patches_per_seq=PATCHES_PER_SEQ,
        seq_length=SEQ_LENGTH,
        normalize=NORMALIZE_BANDS,
        max_invalid_retry=MAX_PATCH_RETRIES,
    )

    train_loader = val_loader = None

    if VAL_SEQUENCES:
        train_loader = create_dataloader(full_ds, shuffle=True, drop_last=True)
        val_ds = TemporalSentinelDronDataset(
            VAL_SEQUENCES,
            patch_size=PATCH_SIZE,
            num_patches_per_seq=PATCHES_PER_SEQ // 2,
            seq_length=SEQ_LENGTH,
            normalize=NORMALIZE_BANDS,
        )
        val_loader = create_dataloader(val_ds, shuffle=False, drop_last=False)
    else:
        n_val = max(1, int(len(full_ds) * TEST_RATIO))
        n_train = len(full_ds) - n_val
        if n_val > 0 and n_train > 0:
            train_sub, val_sub = torch.utils.data.random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(SEED)
            )
            train_loader = create_dataloader(train_sub, shuffle=True,  drop_last=True)
            val_loader   = create_dataloader(val_sub,   shuffle=False, drop_last=False)
        else:
            train_loader = create_dataloader(full_ds, shuffle=True, drop_last=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = DeepSent(
        num_channels  = NUM_CHANNELS,
        feat_channels = FEAT_CHANNELS,
        decoder_layers= 10,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    try:
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=USE_AMP)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDeepSent on {DEVICE}")
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Train sequences    : {len(TRAIN_SEQUENCES)}  |  val: {len(VAL_SEQUENCES)}")
    print(f"  SEQ_LENGTH={SEQ_LENGTH} | PATCH_SIZE={PATCH_SIZE} | BATCH={BATCH_SIZE} | EPOCHS={NUM_EPOCHS}\n")

    best_loss = float("inf")
    train_losses, val_losses = [], []
    train_psnrs,  val_psnrs  = [], []
    _t0_train = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        e_loss, n_batches = 0.0, 0

        for s_seq, d_last in train_loader:
            s_seq  = s_seq.to(DEVICE,  non_blocking=PIN_MEMORY)
            d_last = d_last.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                out  = model(s_seq)
                loss = criterion(out, d_last)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            e_loss   += loss.item()
            n_batches += 1

        avg_loss = e_loss / max(1, n_batches)
        train_losses.append(avg_loss)
        train_psnrs.append(compute_psnr(avg_loss))

        # ── Validation ────────────────────────────────────────────────────────
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vl, vb = 0.0, 0
            with torch.no_grad():
                for s_seq, d_last in val_loader:
                    s_seq  = s_seq.to(DEVICE,  non_blocking=PIN_MEMORY)
                    d_last = d_last.to(DEVICE, non_blocking=PIN_MEMORY)
                    with _autocast():
                        out  = model(s_seq)
                        loss = criterion(out, d_last)
                    vl += loss.item()
                    vb += 1
            if vb > 0:
                val_loss = vl / vb
        val_losses.append(val_loss)
        val_psnrs.append(compute_psnr(val_loss) if np.isfinite(val_loss) else float("nan"))

        lr_now  = optimizer.param_groups[0]["lr"]
        val_str = f"{val_loss:.6f}" if np.isfinite(val_loss) else "N/A"
        print(f"[Epoch {epoch:4d}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}  Val: {val_str}  LR: {lr_now:.2e}")

        scheduler.step()

        save = (avg_loss < best_loss - 1e-6) or \
               (epoch % CHECKPOINT_INTERVAL == 0) or \
               (epoch == NUM_EPOCHS)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
        if save:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Model saved: {MODEL_PATH} (best_loss={best_loss:.6f})")

    # Save the final plot
    try:
        e_arr = np.arange(1, len(train_losses) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(e_arr, train_losses, label="train")
        ax1.plot(e_arr, val_losses,   label="val", linestyle="--")
        ax1.set(title="Loss",  xlabel="Epoch", ylabel="MSE")
        ax1.legend(); ax1.grid(True, alpha=0.4)
        ax2.plot(e_arr, train_psnrs, label="train")
        ax2.plot(e_arr, val_psnrs,   label="val", linestyle="--")
        ax2.set(title="PSNR", xlabel="Epoch", ylabel="dB")
        ax2.legend(); ax2.grid(True, alpha=0.4)
        fig.suptitle(f"DeepSent – Training ({RESOLUTION}, {num_total} dates)")
        fig.tight_layout()
        plot_path = OUT_DIR / f"training_metrics_deepsent_{RESOLUTION}_dates{num_total}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {plot_path}")
    except Exception as exc:
        print(f"[!] Could not save the plot: {exc}")

    _t1_train = time.time()
    print("DeepSent training completed.")
    print(f"  Total training time: {_t1_train - _t0_train:.1f} s  ({(_t1_train - _t0_train)/60:.2f} min)")


# ============================================================
# INFERENCE
# ============================================================
@torch.inference_mode()
def run_inference_on_dir(
    sentinel_infer_dir: Path = SENTINEL_INFER_DIR,
    dron_infer_dir: Path     = DRON_INFER_DIR,
    model_path               = None,
    out_dir: Path            = None,
):
    """
    Runs DeepSent inference for all dates in sentinel_infer_dir.

    Temporal strategy:
        sequence = [training context (up to SEQ_LENGTH-1)] + [inference frame]

    Outputs per date:
        <date>_deepsent_<res>.tif
        <date>_deepsent_metrics.csv
    """
    if out_dir is None:
        out_dir = OUT_DIR / "inferencia_deepsent"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = RESOLUTION

    infer_pairs = build_inference_pairs_from_dir(
        sentinel_infer_dir, dron_infer_dir, target_res=res
    )
    _infer_dates = [d for d, _, _ in infer_pairs]
    print(f"[inference] INFERENCE DATES ({len(_infer_dates)}): {', '.join(_infer_dates)}")

    # Chronological index of training Sentinel images
    train_s_by_date = {Path(s).stem[:8]: s for s, _ in ALL_PAIRS}
    train_s_ordered = [train_s_by_date[d] for d in sorted(train_s_by_date)]

    load_path = model_path if model_path is not None else MODEL_PATH

    # Load the model only once
    model = DeepSent(
        num_channels  = NUM_CHANNELS,
        feat_channels = FEAT_CHANNELS,
        decoder_layers= 10,
    ).to(DEVICE)
    model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"[inference] Model loaded: {load_path}")

    all_metrics = []

    for date, s_infer_path, d_ref_path in infer_pairs:
        print(f"\n{'='*60}")
        print(f"[inference] Date: {date}  →  {s_infer_path.name}")

        # Context: training frames earlier than the inference date
        max_ctx  = (SEQ_LENGTH - 1) if SEQ_LENGTH is not None else None
        ctx_all  = [p for p in train_s_ordered if Path(p).stem[:8] < date]
        context  = ctx_all[-max_ctx:] if max_ctx is not None else ctx_all
        sequence = context + [str(s_infer_path)]
        ctx_dates = [Path(p).stem[:8] for p in context]
        print(f"           Context ({len(context)} frames): {ctx_dates} + [{date}]  (T={len(sequence)})")

        # ── Read and preprocess ──────────────────────────────────────────────
        arrs        = []
        ref_profile = None
        for sp in sequence:
            with rasterio.open(sp) as src:
                if str(sp) == str(s_infer_path):
                    ref_profile = src.profile.copy()
                arr = src.read().astype(np.float32)
                if src.nodata is not None:
                    arr = np.where(arr == src.nodata, np.nan, arr)
                arr = clip_to_index_range(arr)
                if NORMALIZE_BANDS:
                    for b in range(arr.shape[0]):
                        v, mask = arr[b], np.isfinite(arr[b])
                        arr[b] = (v - np.nanmean(v[mask])) / (np.nanstd(v[mask]) + 1e-6) \
                                 if np.any(mask) else np.zeros_like(v)
                arrs.append(np.nan_to_num(arr, nan=0.0))

        s_t = torch.from_numpy(
            np.stack(arrs, axis=0)
        ).unsqueeze(0).to(DEVICE)                      # (1, T, C, H, W)

        with _autocast():
            out = model(s_t)                           # (1, C, H, W)

        out_np = clip_to_index_range(
            out.squeeze(0).cpu().numpy().astype(np.float32)
        )
        out_np = np.nan_to_num(out_np, nan=-9999.0)

        # ── Write GeoTIFF ────────────────────────────────────────────────────
        out_tif = out_dir / f"{date}_deepsent_{res}.tif"
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
                dst.set_band_description(i + 1, f"Band_{i+1}")
        print(f"  → Saved: {out_tif}")

        # ── Evaluation ───────────────────────────────────────────────────────
        if d_ref_path is not None:
            try:
                pred_arr, pred_desc = read_raster(out_tif)
                ref_arr,  ref_desc  = read_raster(d_ref_path)
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
                print(f"SID  (mean): {sid_mean:.6f}" if np.isfinite(sid_mean)
                      else "SID: no valid data")
                print(f"ERGAS: {ergas_val:.3f}" if np.isfinite(ergas_val)
                      else "ERGAS: no valid data")

                csv_out = out_dir / f"{date}_deepsent_metrics.csv"
                save_csv(stats, csv_out, descriptions)
                print(f"  → Metrics: {csv_out}")

                all_metrics.append({"date": date, "SAM": sam_mean,
                                    "SID": sid_mean, "ERGAS": ergas_val})
            except Exception as exc:
                print(f"  [!] Evaluation error for {date}: {exc}")
        else:
            print(f"  [!] No UAV reference for {date}.")

    # ── Global summary CSV ────────────────────────────────────────────────────
    if all_metrics:
        import csv
        summary_csv = out_dir / f"resumen_inferencia_deepsent_{res}.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "SAM", "SID", "ERGAS"])
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[inference] Global summary: {summary_csv}")

    print("\n[inference] DeepSent – Processing completed.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) Train DeepSent
    _t0_train = time.time()
    train_deepsent()
    _t1_train = time.time()
    print(f"\n[TIMER] Total training time (wall-clock): {_t1_train - _t0_train:.1f} s  "
          f"({(_t1_train - _t0_train)/60:.2f} min)")

    # 2) Run inference on sentinel_inferencia/
    _t0_infer = time.time()
    run_inference_on_dir(
        sentinel_infer_dir=SENTINEL_INFER_DIR,
        dron_infer_dir=DRON_INFER_DIR,
        model_path=MODEL_PATH,
    )
    _t1_infer = time.time()
    print(f"\n[TIMER] Total inference time (wall-clock)  : {_t1_infer - _t0_infer:.1f} s  "
          f"({(_t1_infer - _t0_infer)/60:.2f} min)")
