#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVDSR - Temporal Very Deep Super-Resolution adapted for:
    - Input : temporal sequences of Sentinel-2 composites
        (NDVI, SAVI, GNDVI, NDWI, NDVIre) previously resampled

    - Target : UAV-derived composite corresponding to the last date
        in the temporal sequence


The model learns a NONLINEAR SPATIOTEMPORAL MAPPING:

[Sentinel_t1, Sentinel_t2, ..., Sentinel_tT] --> UAV_tT

Architecture (num_layers = 20):

    1. CNN Encoder: 10 Conv2D layers (C→64, followed by 64→64 with ReLU),
    applied INDEPENDENTLY to each temporal frame.

    2. Bidirectional ConvLSTM: Two independent ConvLSTM cells process the sequence
    in FORWARD (t=0→T) and BACKWARD (t=T→0) directions. Each cell is modulated through FiLM using Julian day embeddings.
    
    3. Fusion: cat(h_fwd[t], h_bwd[t]) → Conv1×1 → GroupNorm → h_fused[t]

    4. Temporal Attention: TemporalHiddenAttention assigns weights to the T fused states
    according to their seasonal proximity to the target frame.

    5. CNN Decoder: 10 Conv2D layers (64→64, followed by 64→C with ReLU).

    6. Residual Learning: out = x_T + decoder(h_final_fused)

The network is trained using Mean Squared Error (MSE) between the
predicted output and the UAV composite corresponding to the last
frame of the sequence.
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import math

# New: imports for evaluation and visualization
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


# ---------------- CONFIGURATION ----------------
# Directories where the GeoTIFFs are located (adjust only these two paths)
SENTINEL_DIR = Path(r"Path\to\sentinel_directory")  # Sentinel-2 composites (temporal)
DRON_DIR     = Path(r"Path\to\dron_directory")     # UAV reference images (spatial)

# Target resolution (e.g., "2m", "1m", "1p5m", "0p5m").
# None → the most frequent resolution among the matched pairs is selected automatically.
TARGET_RESOLUTION = "2m"

# Grouping of dates into sequences:
#   "all"  → a single sequence with all dates found (recommended)
#   "year" → one sequence per calendar year
GROUP_BY = "all"

# ── Inference directories (images NOT used for training) ──────────
# sentinel_inference : Sentinel TIFFs to predict (multiple dates, multiple resolutions)
# dron_inference     : UAV reference TIFFs for evaluation
# TARGET_RESOLUTION will be used (same as training) to filter.
SENTINEL_INFER_DIR = Path(r"Path\to\sentinel_inference_directory")
DRON_INFER_DIR     = Path(r"Path\to\dron_inference_directory")


def _parse_tif_stem(stem: str):
    """
    Extracts (date_str, resolution) from names such as:
        20250709_comp_sentinel_2m   →  ('20250709', '2m')
        20250813_comp_dron_1p5m     →  ('20250813', '1p5m')
    Returns (None, None) if it does not match the pattern.
    """
    import re
    m = re.match(r"^(\d{8})_comp_\w+_(\d+(?:p\d+|(?:\.\d+)?)m)$", stem, re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1), m.group(2).lower()


def build_sequences_from_dirs(
    sentinel_dir: Path,
    dron_dir: Path,
    target_res: str = None,
    group_by: str = "all",
):
    """
    Scans sentinel_dir and dron_dir, matches by date + resolution, and
    builds temporally ordered sequences.

    Returns a list of lists: [ [(s_path, d_path), ...], ... ]
    """
    # 1. Index files by (date, resolution)
    def index_dir(directory: Path):
        idx = {}
        for p in sorted(directory.glob("*.tif")):
            date, res = _parse_tif_stem(p.stem)
            if date and res:
                idx[(date, res)] = p
        return idx

    s_idx = index_dir(sentinel_dir)
    d_idx = index_dir(dron_dir)

    # 2. Choose resolution if it was not specified
    common_keys = set(s_idx) & set(d_idx)
    if not common_keys:
        raise FileNotFoundError(
            f"No matching files were found in:\n"
            f"  sentinel: {sentinel_dir}\n  dron: {dron_dir}\n"
            "Check that the names follow the pattern YYYYMMDD_comp_<sensor>_<res>.tif"
        )

    res_to_use = target_res
    if res_to_use is None:
        from collections import Counter as _Counter
        res_to_use = _Counter(res for _date, res in common_keys).most_common(1)[0][0]
        print(f"[auto] Resolution selected automatically: {res_to_use}")

    # 3. Filter pairs with the chosen resolution and sort by date
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

    # 4. Group into sequences
    if group_by == "year":
        from collections import defaultdict as _dd
        by_year = _dd(list)
        for date, sp, dp in pairs:
            by_year[date[:4]].append((str(sp), str(dp)))
        sequences = [seq for seq in by_year.values() if seq]
    else:  # "all"
        sequences = [[(str(sp), str(dp)) for _, sp, dp in pairs]]

    return sequences


def build_inference_pairs_from_dir(
    sentinel_infer_dir: Path,
    dron_infer_dir: Path,
    target_res: str,
):
    """
    Scans sentinel_infer_dir for TIFFs matching target_res and matches them
    with TIFFs from dron_infer_dir having the same date and resolution.

    Returns a list of (date_str, sentinel_path, dron_path_or_None) ordered
    chronologically. dron_path_or_None is None if no reference exists.
    """
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
            f"  {sentinel_infer_dir}\n"
            f"Available resolutions: {sorted({r for _, r in s_idx})}"
        )

    print(f"[inference] {len(pairs)} dates found for resolution '{target_res}':")
    for date, sp, dp in pairs:
        ref_str = dp.name if dp else "(no reference)"
        print(f"             {date}  sentinel={sp.name}  dron_ref={ref_str}")

    return pairs


# Automatic sequence detection
TRAIN_SEQUENCES = build_sequences_from_dirs(
    SENTINEL_DIR, DRON_DIR,
    target_res=TARGET_RESOLUTION,
    group_by=GROUP_BY,
)

# Temporal window length seen by the model at each step.
# With SEQ_LENGTH=4 and 14 dates, 11 sliding windows are generated
# (much more data), and the model always sees exactly T=4 frames,
# ensuring consistency between training and inference.
SEQ_LENGTH = 4  # None = full sequence (not recommended with few dates)

# Split sequences into train/test
TEST_RATIO = 0.2
ALL_SEQUENCES = TRAIN_SEQUENCES.copy()
# For legacy PAIRS (resolution, etc.)
TRAIN_PAIRS = [pair for seq in TRAIN_SEQUENCES for pair in seq]
ALL_PAIRS = TRAIN_PAIRS.copy()
SEED = 2025
random.seed(SEED)
random.shuffle(ALL_PAIRS)
num_total = len(ALL_PAIRS)

_train_dates = sorted({Path(s).stem[:8] for s, _d in ALL_PAIRS})
print(f"[config] Training dates ({len(_train_dates)}): {', '.join(_train_dates)}")

# Extract resolution automatically from file names (e.g., '0p5m', '0.5m', '2m')
import re
from collections import Counter

def extract_resolution_from_string(s: str):
    if not s:
        return None
    m = re.search(r"(\d+(?:[.,]\d+|p\d+)?m)", s, flags=re.IGNORECASE)
    if not m:
        return None
    res = m.group(1).lower()
    res = res.replace('p', '.').replace(',', '.')
    return res


def get_resolution_from_pairs(pairs):
    candidates = []
    for s_path, d_path in pairs:
        for p in (s_path, d_path):
            try:
                stem = Path(p).stem
            except Exception:
                stem = str(p)
            r = extract_resolution_from_string(stem)
            if r:
                candidates.append(r)
            else:
                r = extract_resolution_from_string(str(p))
                if r:
                    candidates.append(r)
    if not candidates:
        return 'unknown'
    return Counter(candidates).most_common(1)[0][0]

RESOLUTION = get_resolution_from_pairs(ALL_PAIRS)

# Train/val split at the SEQUENCE level (not at the individual pair level)
random.seed(SEED)
random.shuffle(ALL_SEQUENCES)
num_total_seqs = len(ALL_SEQUENCES)
num_val_seqs = max(1, int(num_total_seqs * TEST_RATIO)) if num_total_seqs > 1 else 0
VAL_SEQUENCES   = ALL_SEQUENCES[:num_val_seqs]
TRAIN_SEQUENCES = ALL_SEQUENCES[num_val_seqs:]

# Pair aliases for compatibility (resolution, inference, etc.)
num_total = sum(len(s) for s in ALL_SEQUENCES)
VAL_PAIRS   = [pair for seq in VAL_SEQUENCES   for pair in seq]
TRAIN_PAIRS = [pair for seq in TRAIN_SEQUENCES for pair in seq]

# Folder to save the model and outputs
OUT_DIR = Path(r"D:\Nueva carpeta\2025\Fusion de datos\Articulo 2\vdsr_model_bi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dynamic model name that integrates resolution and number of dates
MODEL_FILENAME = f"tvdsr_bi_res{RESOLUTION}_dates{num_total}.pth"
MODEL_PATH = OUT_DIR / MODEL_FILENAME

# Patch size for training
PATCH_SIZE = 96
PATCHES_PER_SEQ = 200        # patches per sliding window and epoch
                              # with 11 windows → ~2200 samples/epoch
BATCH_SIZE = 8               # reduced because each item is now T frames
NUM_EPOCHS = 500
LEARNING_RATE = 0.0003
GRAD_CLIP_NORM = 0.5
LR_STEP_SIZE = 50
LR_GAMMA = 0.5
CHECKPOINT_INTERVAL = 5
MAX_PATCH_RETRIES = 20
NUM_WORKERS = 4
INDEX_MIN = -1.0
INDEX_MAX = 1.0
NORMALIZE_BANDS = False

# Number of bands (in your case: 5 indices)
NUM_CHANNELS = 5
# Hidden channels of the ConvLSTM
LSTM_HIDDEN = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True


def clip_to_index_range(arr: np.ndarray) -> np.ndarray:
    """
    Limits index values to the physical window [-1, 1]
    while preserving NaNs.
    """
    return np.clip(arr, INDEX_MIN, INDEX_MAX)


def set_seed(seed: int = SEED):
    """
    Sets the random seeds to improve reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ── Utility: day of year ─────────────────────────────────────────────────
def date_to_julian(date_str: str) -> float:
    """
    Converts YYYYMMDD → normalized day of year ∈ [0.0, 1.0).
    Allows comparison of seasonal proximity between frames from different years.
    Example: 20240803 (August) and 20250709 (July) are close in the annual cycle.
    """
    from datetime import datetime
    dt = datetime.strptime(str(date_str), "%Y%m%d")
    return (dt.timetuple().tm_yday - 1) / 365.0


JULIAN_EMBED_DIM = 16   # dimension of the sinusoidal julian-day embedding


# ---------------- MODELOS ----------------

class JulianDayEmbedding(nn.Module):
    """
    Cyclic sinusoidal encoding of the day of year.

    Converts a normalized day ∈ [0, 1) into a vector of size embed_dim using
    multiple sin/cos harmonics (1/year, 2/year, 4/year, ...).

    Because it is sinusoidal and cyclic, it captures that December and January
    are seasonally close even though numerically they are at opposite ends of the range.
    It also makes July 2024 and July 2025 produce identical embeddings,
    which helps generalization across years.

    Input  : (B,) float normalized [0, 1)
    Output : (B, embed_dim)
    """

    def __init__(self, embed_dim: int = JULIAN_EMBED_DIM):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim debe ser par"
        self.embed_dim = embed_dim
        freqs = torch.arange(1, embed_dim // 2 + 1, dtype=torch.float32) * 2.0 * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, day_norm: torch.Tensor) -> torch.Tensor:
        """day_norm: (B,) → (B, embed_dim)"""
        x      = day_norm.unsqueeze(-1)           # (B, 1)
        angles = x * self.freqs.unsqueeze(0)      # (B, embed_dim//2)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, embed_dim)


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell: replaces the matrix multiplications of the i/f/o/g gates with
    2-D convolutions, preserving the spatial topology of the feature maps.

    References: Shi et al. "Convolutional LSTM Network: A Machine
    Learning Approach for Precipitation Nowcasting", NeurIPS 2015.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        julian_embed_dim: int = JULIAN_EMBED_DIM,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # A single convolution that generates the 4 concatenated gates
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )
        # FiLM: Feature-wise Linear Modulation guided by the julian day.
        # The day embedding generates (scale, shift) for each of the 4 gates.
        # Initialized to zero → identity behavior at the start of training.
        self.film = nn.Linear(julian_embed_dim, 2 * 4 * hidden_channels, bias=True)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # Initialization: forget-gate bias = 1 (reduces vanishing gradient)
        nn.init.orthogonal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # forget gate bias → 1
        with torch.no_grad():
            self.conv.bias[hidden_channels: 2 * hidden_channels].fill_(1.0)

    def init_hidden(self, batch_size: int, height: int, width: int, device):
        """Returns (h, c) initialized to zero."""
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        julian_emb: torch.Tensor | None = None,
    ):
        """
        x          : (B, in_channels,     H, W)
        h          : (B, hidden_channels, H, W)  previous hidden state
        c          : (B, hidden_channels, H, W)  previous cell state
        julian_emb : (B, julian_embed_dim)        julian-day embedding (optional)
        Returns: (h_new, c_new)
        """
        combined = torch.cat([x, h], dim=1)           # (B, in+hidden, H, W)
        gates = self.conv(combined)                    # (B, 4*hidden, H, W)

        # ── FiLM: modulation by julian day ──────────────────────────────
        if julian_emb is not None:
            film_params = self.film(julian_emb)            # (B, 2*4*hidden)
            scale, shift = film_params.chunk(2, dim=-1)    # each (B, 4*hidden)
            scale = scale.view(x.size(0), -1, 1, 1) + 1.0 # scale ≈ 1 at start
            shift = shift.view(x.size(0), -1, 1, 1)
            gates = gates * scale + shift

        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class TemporalHiddenAttention(nn.Module):
    """
    Temporal Attention Mechanism
    Attention over the hidden states {h1, ..., hT} generated by the fused
    Bidirectional ConvLSTM.

    The module assigns adaptive weights to the T fused hidden states
(forward + backward) according to the SEASONAL PROXIMITY of each
observation to the target frame (the last frame in the ordered sequence).

This mechanism complements the Bidirectional ConvLSTM. While the
BiConvLSTM captures pixel-level spatiotemporal dependencies throughout
the sequence, the attention module determines the relative contribution
of each time step to the final reconstruction.

    Inputs:
        h_stack     : (B, T, lstm_hidden, H, W)  fused hidden states from all temporal steps
        julian_embs : (B, T, embed_dim)           Julian day embeddings associated with each frame
    Output:
        (B, lstm_hidden, H, W)
    """

    def __init__(self, lstm_hidden: int, julian_embed_dim: int = JULIAN_EMBED_DIM):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(2 * julian_embed_dim, julian_embed_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(julian_embed_dim, 1, bias=True),
        )
        for m in self.score_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(
        self,
        h_stack: torch.Tensor,
        julian_embs: torch.Tensor,
    ) -> torch.Tensor:
        B, T, hidden, H, W = h_stack.shape
        emb_target  = julian_embs[:, -1:, :].expand(B, T, -1)       # (B, T, embed_dim)
        score_input = torch.cat([julian_embs, emb_target], dim=-1)   # (B, T, 2*embed_dim)
        scores  = self.score_mlp(score_input)                         # (B, T, 1)
        weights = torch.softmax(scores, dim=1)                        # Σ α_t = 1
        weights = weights.unsqueeze(-1).unsqueeze(-1)                 # (B, T, 1, 1, 1)
        return (h_stack * weights).sum(dim=1)                         # (B, lstm_hidden, H, W)


class SpectralHiddenAttention(nn.Module):
    """
    Lightweight spectral attention (SE-type) over the fused hidden context.

    Input:
        h_stack : (B, T, C, H, W)
    Output:
        (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, h_stack: torch.Tensor) -> torch.Tensor:
        h_mean = h_stack.mean(dim=1)                         # (B, C, H, W)
        gap = h_mean.mean(dim=(2, 3))                        # (B, C)
        gmp = h_mean.amax(dim=(2, 3))                        # (B, C)
        weights = torch.sigmoid(self.mlp(gap) + self.mlp(gmp)).unsqueeze(-1).unsqueeze(-1)
        return h_mean * weights                              # (B, C, H, W)


class ParallelAttentionFusion(nn.Module):
    """
    Fusion with learnable gates for parallel branches.
    Fusion with 2 branches (temporal, spectral).
    """

    def __init__(self):
        super().__init__()
        self.branch_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def forward(
        self,
        temporal_feat: torch.Tensor,
        spectral_feat: torch.Tensor,
    ) -> torch.Tensor:
        w = torch.softmax(self.branch_logits, dim=0)
        return w[0] * temporal_feat + w[1] * spectral_feat


class TVDSRBi(nn.Module):
    """
    Temporal Very Deep Super-Resolution with BiConvLSTM.

    Processes temporal sequences through:
    - Spatial encoding of each frame (10-layer CNN encoder)
    - Bidirectional temporal processing (forward+backward ConvLSTM with FiLM)
        - Parallel attention branches:
            1) temporal (seasonal), 2) spectral (lightweight)
        - Fusion by learnable gates for 2 branches
    - Spatiotemporal decoding (10-layer CNN decoder)
    - Residual learning from the last frame
    """

    def __init__(
        self,
        num_channels: int = 5,
        num_layers: int = 20,
        lstm_hidden: int = 64,
        num_lstm_layers: int = 1,
        julian_embed_dim: int = JULIAN_EMBED_DIM,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        self.julian_embed_dim = julian_embed_dim

        # Sinusoidal embedding of the julian day
        self.julian_emb = JulianDayEmbedding(julian_embed_dim)

        # Shared encoder
        enc_depth = num_layers // 2
        enc_layers = [
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]
        for _ in range(enc_depth - 1):
            enc_layers += [
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*enc_layers)

        # ConvLSTM forward/backward
        self.lstm_fwd_cells = nn.ModuleList()
        self.lstm_bwd_cells = nn.ModuleList()
        for _ in range(num_lstm_layers):
            self.lstm_fwd_cells.append(
                ConvLSTMCell(64, lstm_hidden, julian_embed_dim=julian_embed_dim)
            )
            self.lstm_bwd_cells.append(
                ConvLSTMCell(64, lstm_hidden, julian_embed_dim=julian_embed_dim)
            )

        # Temporal fusion per timestep
        self.fusion = nn.Conv2d(2 * lstm_hidden, lstm_hidden, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.fusion.weight, nonlinearity="relu")
        nn.init.zeros_(self.fusion.bias)
        self.fusion_norm = nn.GroupNorm(num_groups=8, num_channels=lstm_hidden)

        # Parallel attention branches (low cost)
        self.temporal_attn = TemporalHiddenAttention(lstm_hidden, julian_embed_dim)
        self.spectral_attn = SpectralHiddenAttention(lstm_hidden, reduction=8)
        self.attn_fusion = ParallelAttentionFusion()

        # Decoder
        dec_depth = num_layers - enc_depth
        dec_layers: list[nn.Module] = []
        for _ in range(dec_depth - 1):
            dec_layers += [
                nn.Conv2d(lstm_hidden, lstm_hidden, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
        dec_layers.append(
            nn.Conv2d(lstm_hidden, num_channels, kernel_size=3, padding=1, bias=True)
        )
        self.decoder = nn.Sequential(*dec_layers)

        # He initialization for Conv2d (except ConvLSTM, which is initialized in its class)
        init_modules = (
            list(self.encoder.modules())
            + list(self.decoder.modules())
        )
        for m in init_modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_seq: torch.Tensor,
        julian_days: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_seq       : (B, T, C, H, W)
        julian_days : (B, T) float in [0, 1) normalized day of year.
                      If None it is omitted (backward-compatible).
        """
        B, T, C, H, W = x_seq.shape

        # 1. Sort frames by julian day (order independence)
        if julian_days is not None:
            sort_idx = torch.argsort(julian_days, dim=1)
            idx_img = sort_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(B, T, C, H, W)
            x_seq = torch.gather(x_seq, 1, idx_img)
            julian_days = torch.gather(julian_days, 1, sort_idx)

        # 2. Julian-day embeddings
        if julian_days is not None:
            emb_flat = self.julian_emb(julian_days.view(B * T))
            julian_embs = emb_flat.view(B, T, self.julian_embed_dim)
        else:
            julian_embs = None

        # 3. Encode all frames
        feats = [self.encoder(x_seq[:, t]) for t in range(T)]

        # 4. Forward ConvLSTM
        states_fwd = [cell.init_hidden(B, H, W, x_seq.device) for cell in self.lstm_fwd_cells]
        h_fwd = []
        for t in range(T):
            feat = feats[t]
            j_emb_t = julian_embs[:, t] if julian_embs is not None else None
            for layer_idx, cell in enumerate(self.lstm_fwd_cells):
                h, c = states_fwd[layer_idx]
                h, c = cell(feat, h, c, julian_emb=j_emb_t)
                states_fwd[layer_idx] = (h, c)
                feat = h
            h_fwd.append(states_fwd[-1][0])

        # 5. Backward ConvLSTM
        states_bwd = [cell.init_hidden(B, H, W, x_seq.device) for cell in self.lstm_bwd_cells]
        h_bwd = [None] * T
        for t in reversed(range(T)):
            feat = feats[t]
            j_emb_t = julian_embs[:, t] if julian_embs is not None else None
            for layer_idx, cell in enumerate(self.lstm_bwd_cells):
                h, c = states_bwd[layer_idx]
                h, c = cell(feat, h, c, julian_emb=j_emb_t)
                states_bwd[layer_idx] = (h, c)
                feat = h
            h_bwd[t] = states_bwd[-1][0]

        # 6. Fusion per timestep
        h_fused = []
        for t in range(T):
            h_cat = torch.cat([h_fwd[t], h_bwd[t]], dim=1)
            h_fused.append(self.fusion_norm(self.fusion(h_cat)))

        # 7. Parallel attention (temporal + spectral) and fusion
        h_tensor = torch.stack(h_fused, dim=1)

        if julian_embs is not None:
            h_temporal = self.temporal_attn(h_tensor, julian_embs)
        else:
            h_temporal = h_fused[-1]

        h_spectral = self.spectral_attn(h_tensor)
        h_final = self.attn_fusion(h_temporal, h_spectral)

        # 8. Residual prediction + skip
        x_last = x_seq[:, -1]
        residual = self.decoder(h_final)
        return x_last + residual



# ---------------- TEMPORAL DATASET ----------------
class TemporalSentinelDronDataset(Dataset):
    """
    Sentinel–UAV temporal patch dataset.

    Each item returns:
        s_seq (T, C, H, W)  — Sentinel patch sequence in temporal order
        d_last (C, H, W)    — UAV patch from the LAST frame (target)

    All patches from the same sample are extracted from the same spatial
    location (r0, c0) to ensure spatiotemporal consistency.

    Parameter seq_length:
        None  → use the full sequence (all frames)
        int N → sliding windows of N consecutive frames
    """

    def __init__(
        self,
        sequences,                  # lista de listas de (s_path, d_path)
        patch_size: int = 64,
        num_patches_per_seq: int = 200,
        seq_length=None,
        normalize: bool = True,
        max_invalid_retry: int = MAX_PATCH_RETRIES,
    ):
        self.patch_size  = patch_size
        self.seq_length  = seq_length
        self.normalize   = normalize
        self.max_retry   = max_invalid_retry

        # ── Pre-load all images into RAM ─────────────────────────
        # self.sequences: list of lists of (s_arr, d_arr) already normalized/scaled
        self.sequences: list[list[tuple]] = []
        for seq in sequences:
            loaded_seq = []
            for s_path, d_path in seq:
                with rasterio.open(s_path) as ss, rasterio.open(d_path) as ds:
                    s_arr = ss.read().astype(np.float32)   # (C, H, W)
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
                                v = arr[b]
                                mask = np.isfinite(v)
                                if np.any(mask):
                                    mu = np.nanmean(v[mask])
                                    sd = np.nanstd(v[mask]) + 1e-6
                                    arr[b] = (v - mu) / sd

                    loaded_seq.append((s_arr, d_arr, date_to_julian(Path(s_path).stem[:8])))
            self.sequences.append(loaded_seq)

        if not self.sequences:
            raise ValueError("No hay secuencias Sentinel-Dron cargadas.")

        self.H = self.sequences[0][0][0].shape[1]
        self.W = self.sequences[0][0][0].shape[2]

        # ── Build the window index: (seq_idx, start_t) ─────────────
        self.windows: list[tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            T = len(seq)
            win = seq_length if seq_length is not None else T
            if win > T:
                raise ValueError(
                    f"seq_length={win} > longitud de secuencia={T}. "
                    "Reduce SEQ_LENGTH o añade más fechas."
                )
            for start_t in range(T - win + 1):
                self.windows.append((seq_idx, start_t))

        self.num_patches = num_patches_per_seq * len(self.windows)

    def __len__(self) -> int:
        return self.num_patches

    def __getitem__(self, idx: int):
        ps = self.patch_size
        max_row = self.H - ps
        max_col = self.W - ps
        if max_row <= 0 or max_col <= 0:
            raise ValueError("Image smaller than PATCH_SIZE. Reduce PATCH_SIZE.")

        win = self.seq_length   # None if the full sequence is used

        for _ in range(self.max_retry):
            # Randomly choose a window and a spatial position
            win_idx  = random.randint(0, len(self.windows) - 1)
            seq_idx, start_t = self.windows[win_idx]
            seq = self.sequences[seq_idx]
            T = len(seq) if win is None else win

            r0 = random.randint(0, max_row)
            c0 = random.randint(0, max_col)

            s_patches = []
            d_last    = None
            valid = True

            frames = seq[start_t: start_t + T]
            frame_julians = []
            for t_idx, (s_arr, d_arr, julian_float) in enumerate(frames):
                frame_julians.append(julian_float)
                sp = s_arr[:, r0:r0+ps, c0:c0+ps]
                dp = d_arr[:, r0:r0+ps, c0:c0+ps]

                if not np.isfinite(sp).any() or not np.isfinite(dp).any():
                    valid = False
                    break

                sp = np.nan_to_num(sp, nan=0.0)
                dp = np.nan_to_num(dp, nan=0.0)
                s_patches.append(sp)
                if t_idx == len(frames) - 1:
                    d_last = dp

            if not valid or d_last is None:
                continue

            # s_seq: (T, C, ps, ps) ; d_last: (C, ps, ps) ; julian_t: (T,)
            s_seq    = torch.from_numpy(np.stack(s_patches, axis=0)).float()  # (T, C, H, W)
            d_t      = torch.from_numpy(d_last).float()                        # (C, H, W)
            julian_t = torch.tensor(frame_julians, dtype=torch.float32)        # (T,)
            return s_seq, d_t, julian_t

        raise RuntimeError("A valid temporal patch could not be extracted.")


# ── Compatibility alias ─────────────────────────────────────────────────
SentinelDronDataset = TemporalSentinelDronDataset


# ---------------- TRAINING ----------------
def create_dataloader(dataset, shuffle=True, drop_last=True):
    """
    Builds a DataLoader optimized for GPU (pin_memory) when available.
    """
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
        pin_memory=PIN_MEMORY
    )
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def compute_psnr(mse, data_range=INDEX_MAX - INDEX_MIN):
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((data_range ** 2) / mse)



try:
    plt.ion()
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def init_training_plot():
    """
    Initializes an interactive figure to monitor Loss and PSNR.
    Returns (fig, ax_loss, ax_psnr) or None if matplotlib is unavailable.
    """
    if not HAS_MPL:
        return None
    try:
        plt.ion()
    except Exception:
        pass
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.set_title("TVDSRBi - Training evolution")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax2.set_ylabel("PSNR (dB)", color="tab:orange")
    fig.tight_layout()
    return fig, ax1, ax2


def update_training_plot(handles, epochs, train_losses, train_psnrs, eval_losses=None, eval_psnrs=None):
    """
    Refreshes the interactive figure with the accumulated training and validation history.
    """
    if handles is None:
        return
    try:
        fig, ax1, ax2 = handles
    except Exception:
        return

    ax1.cla()
    ax2.cla()
    ax1.set_title("TVDSRBi - Training evolution")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax2.set_ylabel("PSNR (dB)", color="tab:orange")

    # Plot loss
    if hasattr(epochs, '__len__') and len(epochs) > 0 and len(train_losses) > 0:
        try:
            ax1.plot(epochs, train_losses, color="tab:blue", label="Loss (train)")
        except Exception:
            pass
    if eval_losses is not None and len(eval_losses) > 0:
        try:
            ax1.plot(epochs, eval_losses, color="tab:green", linestyle="--", label="Loss (val)")
        except Exception:
            pass

    ax1.grid(True, linestyle="--", alpha=0.4)

    # Plot PSNR
    if len(train_psnrs) > 0:
        try:
            ax2.plot(epochs, train_psnrs, color="tab:orange", label="PSNR (train)")
        except Exception:
            pass
    if eval_psnrs is not None and len(eval_psnrs) > 0:
        try:
            ax2.plot(epochs, eval_psnrs, color="tab:red", linestyle="--", label="PSNR (val)")
        except Exception:
            pass

    try:
        fig.legend(loc="upper right")
    except Exception:
        pass

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass


def train_tvdsr():
    """
    Trains the TVDSR model on temporal sequences of Sentinel–UAV pairs.
    Model input  : s_seq (B, T, C, H, W)
    Target       : d_last (B, C, H, W)  — UAV patch from the last frame
    """
    set_seed(SEED)
    if len(TRAIN_SEQUENCES) == 0:
        raise ValueError(
            "No training sequences available. "
            "Make sure that TRAIN_SEQUENCES contains a sufficient number of dates."
        )

    handles = init_training_plot()

    # ── Training dataset ────────────────────────────────────────────
    full_dataset = TemporalSentinelDronDataset(
        TRAIN_SEQUENCES,
        patch_size=PATCH_SIZE,
        num_patches_per_seq=PATCHES_PER_SEQ,
        seq_length=SEQ_LENGTH,
        normalize=NORMALIZE_BANDS,
        max_invalid_retry=MAX_PATCH_RETRIES,
    )

    train_loader = None
    val_loader   = None

    if len(VAL_SEQUENCES) > 0:
        train_loader = create_dataloader(full_dataset, shuffle=True, drop_last=True)
        val_dataset = TemporalSentinelDronDataset(
            VAL_SEQUENCES,
            patch_size=PATCH_SIZE,
            num_patches_per_seq=PATCHES_PER_SEQ // 2,
            seq_length=SEQ_LENGTH,
            normalize=NORMALIZE_BANDS,
            max_invalid_retry=MAX_PATCH_RETRIES,
        )
        val_loader = create_dataloader(val_dataset, shuffle=False, drop_last=False)
    else:
        total_patches = len(full_dataset)
        val_count  = max(1, int(total_patches * TEST_RATIO)) if total_patches > 1 else 0
        train_count = total_patches - val_count
        if val_count > 0 and train_count > 0:
            train_sub, val_sub = torch.utils.data.random_split(
                full_dataset,
                [train_count, val_count],
                generator=torch.Generator().manual_seed(SEED),
            )
            train_loader = create_dataloader(train_sub, shuffle=True, drop_last=True)
            val_loader   = create_dataloader(val_sub,   shuffle=False, drop_last=False)
        else:
            train_loader = create_dataloader(full_dataset, shuffle=True, drop_last=True)

    # ── Modelo TVDSRBi ───────────────────────────────────────────────────────
    model = TVDSRBi(
        num_channels=NUM_CHANNELS,
        num_layers=20,
        lstm_hidden=LSTM_HIDDEN,
        num_lstm_layers=1,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    try:
        scaler = torch.amp.GradScaler(device_type='cuda', enabled=USE_AMP)
    except Exception:
        scaler = torch.amp.GradScaler() if USE_AMP else torch.amp.GradScaler(enabled=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TVDSRBi en {DEVICE} | parámetros: {total_params:,} | "
          f"secuencias_train={len(TRAIN_SEQUENCES)} | val_seqs={len(VAL_SEQUENCES)}")
    best_loss = float("inf")

    train_losses, val_losses = [], []
    train_psnrs,  val_psnrs  = [], []
    _t0_train = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # s_seq: (B, T, C, H, W)  |  d_last: (B, C, H, W)  |  julian_days: (B, T)
        for s_seq, d_last, julian_days in train_loader:
            s_seq       = s_seq.to(DEVICE,       non_blocking=PIN_MEMORY)
            d_last      = d_last.to(DEVICE,      non_blocking=PIN_MEMORY)
            julian_days = julian_days.to(DEVICE, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out  = model(s_seq, julian_days)   # (B, C, H, W)
                loss = criterion(out, d_last)

            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        train_losses.append(avg_loss)
        try:
            train_psnrs.append(compute_psnr(avg_loss))
        except Exception:
            train_psnrs.append(float('nan'))

        # ── Validation ──────────────────────────────────────────────────
        val_loss_epoch = float('nan')
        val_psnr_epoch = float('nan')
        if val_loader is not None:
            model.eval()
            v_loss, v_batches, v_mse = 0.0, 0, 0.0
            with torch.no_grad():
                for s_seq, d_last, julian_days in val_loader:
                    s_seq       = s_seq.to(DEVICE,       non_blocking=PIN_MEMORY)
                    d_last      = d_last.to(DEVICE,      non_blocking=PIN_MEMORY)
                    julian_days = julian_days.to(DEVICE, non_blocking=PIN_MEMORY)
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        out  = model(s_seq, julian_days)
                        loss = criterion(out, d_last)
                    v_loss   += loss.item()
                    v_mse    += torch.mean((out - d_last) ** 2).item()
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

        save_model = (avg_loss < best_loss - 1e-6) or \
                     (epoch % CHECKPOINT_INTERVAL == 0) or \
                     (epoch == NUM_EPOCHS)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
        if save_model:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Model saved: {MODEL_PATH} (best_loss={best_loss:.6f})")

    # Save final plots
    epochs = list(range(1, len(train_losses) + 1))
    if HAS_MPL:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        import numpy as _np
        e = _np.array(epochs)
        t_losses = _np.array(train_losses, dtype=float)
        v_losses = _np.array(val_losses, dtype=float)
        if t_losses.size > 0:
            plt.plot(e, t_losses, label='train_loss')
        if v_losses.size > 0:
            # align lengths
            plt.plot(e, v_losses[: e.size], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        t_ps = _np.array(train_psnrs, dtype=float)
        v_ps = _np.array(val_psnrs, dtype=float)
        if t_ps.size > 0:
            plt.plot(e, t_ps, label='train_psnr')
        if v_ps.size > 0:
            plt.plot(e, v_ps[: e.size], label='val_psnr')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.grid(True)

        plot_path = OUT_DIR / f"training_metrics_tvdsr_bi_res{RESOLUTION}_dates{num_total}.png"
        plt.suptitle("TVDSRBi - Training evolution", y=1.02)
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Final metric plot saved to: {plot_path}")

    _t1_train = time.time()
    print("Training TVDSRBi finished.")
    print(f"  Total training time: {_t1_train - _t0_train:.1f} s  ({(_t1_train - _t0_train)/60:.2f} min)")


# compatibility alias for other scripts
train_vdsr = train_tvdsr


# ---------------- INFERENCE ----------------
@torch.inference_mode()
def apply_tvdsr_to_full_image(
    sentinel_sequence,          # list of Sentinel paths (chronological order)
    out_path,
    model_path=None,
    evaluate=True,
    csv_out=None,
    reference_dron_path=None,   # path to the UAV of the last date (for metrics)
):
    """
    Applies the trained TVDSR to a temporal sequence of Sentinel composites.

    sentinel_sequence : list of str/Path with Sentinel TIFFs in temporal order.
                        The model predicts the UAV of the LAST date.
    out_path          : output path for the predicted GeoTIFF.
    evaluate          : if True and a reference can be resolved, it computes metrics.
    reference_dron_path: explicit path to the reference UAV. If None, it tries to
                        infer it automatically from ALL_PAIRS.
    """
    model = TVDSRBi(
        num_channels=NUM_CHANNELS,
        num_layers=20,
        lstm_hidden=LSTM_HIDDEN,
        num_lstm_layers=1,
    ).to(DEVICE)
    load_path = model_path if model_path is not None else MODEL_PATH
    model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # ── Read and preprocess the sequence ──────────────────────────────────────
    arrs = []
    ref_profile = None
    for s_path in sentinel_sequence:
        with rasterio.open(s_path) as src:
            if ref_profile is None:
                ref_profile = src.profile.copy()
            arr = src.read().astype(np.float32)   # (C, H, W)
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

    # s_t: (1, T, C, H, W) ; jd_t: (1, T)
    s_np  = np.stack(arrs, axis=0)                              # (T, C, H, W)
    s_t   = torch.from_numpy(s_np).unsqueeze(0).to(DEVICE)     # (1, T, C, H, W)
    jd_np = np.array([date_to_julian(Path(p).stem[:8]) for p in sentinel_sequence],
                     dtype=np.float32)                          # (T,)
    jd_t  = torch.from_numpy(jd_np).unsqueeze(0).to(DEVICE)    # (1, T)

    with torch.amp.autocast('cuda', enabled=USE_AMP):
        out = model(s_t, jd_t)                                 # (1, C, H, W)

    out_np = out.squeeze(0).cpu().numpy().astype(np.float32)   # (C, H, W)
    out_np = clip_to_index_range(out_np)
    out_np = np.nan_to_num(out_np, nan=-9999.0)

    # ── Write GeoTIFF ─────────────────────────────────────────────────────
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
    print(f"Imagen TVDSRBi-SR guardada en: {out_path}")

    # ── Evaluation ───────────────────────────────────────────────────────────
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
                ref_arr,  ref_desc  = read_raster(Path(reference_dron_path))
                stats, ref_means = compute_band_metrics(
                    pred_arr, ref_arr, data_range=INDEX_MAX - INDEX_MIN
                )
                descriptions = pred_desc if any(pred_desc) else ref_desc
                print(format_table(stats, descriptions))

                sam_mean, sam_med = spectral_angle_mapper(pred_arr, ref_arr)
                sid_mean          = spectral_information_divergence(pred_arr, ref_arr)
                ergas_val         = ergas(stats, ref_means, scale_ratio=1.0)

                print(f"SAM  (mean/median) [°]: "
                      f"{sam_mean:.3f} / {sam_med:.3f}" if np.isfinite(sam_mean)
                      else "SAM: no valid data")
                print(f"SID  (mean): {sid_mean:.6f}" if np.isfinite(sid_mean)
                      else "SID: no valid data")
                print(f"ERGAS: {ergas_val:.3f}" if np.isfinite(ergas_val)
                      else "ERGAS: no valid data")

                save_csv(stats, csv_out, descriptions)
                print(f"  → Metrics: {csv_out}")

            except Exception as exc:
                print(f"Could not evaluate the inferred image: {exc}")
        else:
            print(
                "No automatic reference was found. "
                "Use reference_dron_path to provide one manually."
            )


# compatibility alias
apply_vdsr_to_full_image = apply_tvdsr_to_full_image


# ---------------- DIRECTORY INFERENCE ----------------
def run_inference_on_dir(
    sentinel_infer_dir: Path = SENTINEL_INFER_DIR,
    dron_infer_dir: Path     = DRON_INFER_DIR,
    model_path=None,
    out_dir: Path = None,
):
    """
    Runs TVDSR inference over ALL dates found in sentinel_infer_dir
    (filtered by TARGET_RESOLUTION) and evaluates each prediction against
    the reference from dron_infer_dir.

    Temporal strategy:
        input sequence = [training Sentinel (context)] +
                          [inference Sentinel (frame to predict)]
    The model predicts the UAV of the last frame (the inference date).

    Outputs per date in out_dir:
        <date>_tvdsr_<res>.tif      → predicted image
        <date>_tvdsr_metrics.csv    → evaluation metrics
    """
    if out_dir is None:
        out_dir = OUT_DIR / "inferencia"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training resolution already resolved in RESOLUTION
    res = RESOLUTION

    # 1. Detect inference dates
    infer_pairs = build_inference_pairs_from_dir(
        sentinel_infer_dir, dron_infer_dir, target_res=res
    )
    _infer_dates = [d for d, _s, _r in infer_pairs]
    print(f"[inference] INFERENCE DATES ({len(_infer_dates)}): {', '.join(_infer_dates)}")

    # 2. Training Sentinel index: {date_YYYYMMDD: path}
    train_sentinel_by_date = {
        Path(s).stem[:8]: s
        for s, _d in ALL_PAIRS
    }
    train_sentinel_paths_ordered = [
        train_sentinel_by_date[d]
        for d in sorted(train_sentinel_by_date)
    ]

    load_path = model_path if model_path is not None else MODEL_PATH

    # 3. Load the model only once for all dates
    model_infer = TVDSRBi(
        num_channels=NUM_CHANNELS,
        num_layers=20,
        lstm_hidden=LSTM_HIDDEN,
        num_lstm_layers=1,
    ).to(DEVICE)
    model_infer.load_state_dict(
        torch.load(load_path, map_location=DEVICE, weights_only=True)
    )
    model_infer.eval()
    print(f"[inference] Model loaded from: {load_path}")

    all_metrics = []  # for global summary CSV

    for date, s_infer_path, d_ref_path in infer_pairs:
        print(f"\n{'='*60}")
        print(f"[inference] Date: {date}  →  {s_infer_path.name}")

        # ── Temporal context: only training frames BEFORE the inference date ──────
        # in strict chronological order.
        # At most the last SEQ_LENGTH-1 frames are used so that the total
        # sequence (context + inference frame) has T=SEQ_LENGTH,
        # matching the training setup.
        max_context = (SEQ_LENGTH - 1) if SEQ_LENGTH is not None else None
        context_all = [p for p in train_sentinel_paths_ordered if Path(p).stem[:8] < date]
        if max_context is not None:
            context = context_all[-max_context:]   # last N previous frames
        else:
            context = context_all
        sequence = context + [str(s_infer_path)]
        if context:
            print(f"           Context ({len(context)} frames): {[Path(p).stem[:8] for p in context]} + [{date}]  (T={len(sequence)})")
        else:
            print(f"           No previous context before {date} → single-frame mode (T=1)")

        out_tif = out_dir / f"{date}_tvdsr_bi_{res}.tif"
        csv_out = out_dir / f"{date}_tvdsr_bi_metrics.csv"

        # ── Preprocess the full sequence ────────────────────────────────
        arrs = []
        ref_profile = None
        for sp in sequence:
            with rasterio.open(sp) as src:
                # The geospatial profile is ALWAYS taken from the inference frame
                if str(sp) == str(s_infer_path):
                    ref_profile = src.profile.copy()
                arr = src.read().astype(np.float32)
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

        s_np  = np.stack(arrs, axis=0)                          # (T, C, H, W)
        s_t   = torch.from_numpy(s_np).unsqueeze(0).to(DEVICE) # (1, T, C, H, W)
        jd_np = np.array([date_to_julian(Path(p).stem[:8]) for p in sequence],
                         dtype=np.float32)                      # (T,)
        jd_t  = torch.from_numpy(jd_np).unsqueeze(0).to(DEVICE) # (1, T)

        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model_infer(s_t, jd_t)                   # (1, C, H, W)

        out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
        out_np = clip_to_index_range(out_np)
        out_np = np.nan_to_num(out_np, nan=-9999.0)

        # ── Write GeoTIFF ─────────────────────────────────────────────
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

        # ── Evaluation ───────────────────────────────────────────────────
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

                print(f"SAM  (mean/median) [°]: "
                      f"{sam_mean:.3f} / {sam_med:.3f}" if np.isfinite(sam_mean)
                      else "SAM: no valid data")
                print(f"SID  (mean): {sid_mean:.6f}" if np.isfinite(sid_mean)
                      else "SID: no valid data")
                print(f"ERGAS: {ergas_val:.3f}" if np.isfinite(ergas_val)
                      else "ERGAS: no valid data")

                save_csv(stats, csv_out, descriptions)
                print(f"  → Metrics: {csv_out}")

                all_metrics.append({
                    "date": date, "SAM": sam_mean,
                    "SID": sid_mean, "ERGAS": ergas_val,
                })
            except Exception as exc:
                print(f"  [!] Error in evaluation for {date}: {exc}")
        else:
            print(f"  [!] No UAV reference for {date}; evaluation is skipped.")

    # ── Global summary CSV ────────────────────────────────────────────────
    if all_metrics:
        import csv
        summary_csv = out_dir / f"resumen_inferencia_tvdsr_bi_{res}.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "SAM", "SID", "ERGAS"])
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\n[inference] Global summary saved to: {summary_csv}")

    print("\n[inference] Processing completed.")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # 1) Train TVDSR
    _t0_train = time.time()
    train_tvdsr()
    _t1_train = time.time()
    print(f"\n[TIMER] Total training time (wall-clock): {_t1_train - _t0_train:.1f} s  "
          f"({(_t1_train - _t0_train)/60:.2f} min)")

    # 2) Inference on sentinel_inference dates
    _t0_infer = time.time()
    run_inference_on_dir(
        sentinel_infer_dir=SENTINEL_INFER_DIR,
        dron_infer_dir=DRON_INFER_DIR,
        model_path=MODEL_PATH,
    )
    _t1_infer = time.time()
    print(f"\n[TIMER] Total inference time (wall-clock)  : {_t1_infer - _t0_infer:.1f} s  "
          f"({(_t1_infer - _t0_infer)/60:.2f} min)")

