#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador independiente de métricas para comparar una imagen superresuelta con su referencia.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio

try:
    from openpyxl import Workbook
except Exception:
    Workbook = None


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        nodata = src.nodata
        descriptions = list(src.descriptions)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, descriptions


def global_ssim(x, y, mask):
    """Versión simplificada de SSIM (promedio global)."""
    if not np.any(mask):
        return np.nan
    x_valid = x[mask]
    y_valid = y[mask]
    c1 = (0.01 * 2) ** 2
    c2 = (0.03 * 2) ** 2
    mu_x = np.mean(x_valid)
    mu_y = np.mean(y_valid)
    sigma_x = np.var(x_valid)
    sigma_y = np.var(y_valid)
    sigma_xy = np.mean((x_valid - mu_x) * (y_valid - mu_y))
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denom = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return num / denom if denom != 0 else np.nan


def uiqi(x, y, mask):
    """Universal Image Quality Index."""
    if not np.any(mask):
        return np.nan
    x_valid = x[mask]
    y_valid = y[mask]
    mu_x = np.mean(x_valid)
    mu_y = np.mean(y_valid)
    var_x = np.var(x_valid)
    var_y = np.var(y_valid)
    cov = np.mean((x_valid - mu_x) * (y_valid - mu_y))
    denom = (mu_x ** 2 + mu_y ** 2) * (var_x + var_y)
    if denom == 0:
        return np.nan
    return (4 * mu_x * mu_y * cov) / denom


def compute_band_metrics(pred, ref, data_range):
    if pred.shape != ref.shape:
        raise ValueError(f"Dimensiones distintas: pred {pred.shape} vs ref {ref.shape}")

    stats = []
    ref_means = []
    for b in range(pred.shape[0]):
        p = pred[b]
        r = ref[b]
        mask = np.isfinite(p) & np.isfinite(r)
        if not np.any(mask):
            stats.append(
                dict(
                    band=b + 1,
                    valid_pixels=0,
                    mse=np.nan,
                    rmse=np.nan,
                    mae=np.nan,
                    psnr=np.nan,
                    bias=np.nan,
                    corr=np.nan,
                    ssim=np.nan,
                    uiqi=np.nan,
                )
            )
            ref_means.append(np.nan)
            continue

        diff = p[mask] - r[mask]
        mse = float(np.mean(diff ** 2))
        rmse = math.sqrt(mse)
        mae = float(np.mean(np.abs(diff)))
        bias = float(np.mean(diff))
        psnr = np.inf if mse == 0 else 10.0 * math.log10((data_range ** 2) / mse)

        x = p[mask] - np.mean(p[mask])
        y = r[mask] - np.mean(r[mask])
        denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
        corr = float(np.sum(x * y) / denom) if denom != 0 else np.nan

        stats.append(
            dict(
                band=b + 1,
                valid_pixels=int(mask.sum()),
                mse=mse,
                rmse=rmse,
                mae=mae,
                psnr=psnr,
                bias=bias,
                corr=corr,
                ssim=float(global_ssim(p, r, mask)),
                uiqi=float(uiqi(p, r, mask)),
            )
        )
        ref_means.append(float(np.mean(r[mask])))
    return stats, ref_means


def spectral_angle_mapper(pred, ref):
    """Calcula SAM (media y mediana de ángulos en grados)."""
    bands, h, w = pred.shape
    pred_flat = pred.reshape(bands, -1)
    ref_flat = ref.reshape(bands, -1)
    mask = np.all(np.isfinite(pred_flat), axis=0) & np.all(np.isfinite(ref_flat), axis=0)
    if not np.any(mask):
        return np.nan, np.nan
    p = pred_flat[:, mask]
    r = ref_flat[:, mask]
    dot = np.sum(p * r, axis=0)
    norm_p = np.linalg.norm(p, axis=0)
    norm_r = np.linalg.norm(r, axis=0)
    denom = norm_p * norm_r
    valid = denom > 0
    angles = np.empty_like(dot)
    angles[:] = np.nan
    angles[valid] = np.degrees(np.arccos(np.clip(dot[valid] / denom[valid], -1.0, 1.0)))
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan, np.nan
    return float(np.mean(angles)), float(np.median(angles))


def spectral_information_divergence(pred, ref, eps=1e-6):
    """SID promedio (nat). Convierte cada vector en distribución de probabilidad."""
    bands, h, w = pred.shape
    pred_flat = pred.reshape(bands, -1)
    ref_flat = ref.reshape(bands, -1)
    mask = np.all(np.isfinite(pred_flat), axis=0) & np.all(np.isfinite(ref_flat), axis=0)
    if not np.any(mask):
        return np.nan
    p = pred_flat[:, mask].T  # (N, bands)
    r = ref_flat[:, mask].T
    sid_vals = []
    for p_vec, r_vec in zip(p, r):
        min_val = min(p_vec.min(), r_vec.min())
        if min_val <= 0:
            shift = abs(min_val) + eps
            p_tmp = p_vec + shift
            r_tmp = r_vec + shift
        else:
            p_tmp = p_vec
            r_tmp = r_vec
        p_prob = p_tmp / (np.sum(p_tmp) + eps)
        r_prob = r_tmp / (np.sum(r_tmp) + eps)
        sid = np.sum(p_prob * np.log((p_prob + eps) / (r_prob + eps))) + np.sum(
            r_prob * np.log((r_prob + eps) / (p_prob + eps))
        )
        sid_vals.append(sid)
    if not sid_vals:
        return np.nan
    return float(np.mean(sid_vals))


def ergas(stats, ref_means, scale_ratio=1.0):
    """Calcula ERGAS (menor es mejor)."""
    terms = []
    for stat, mean_ref in zip(stats, ref_means):
        if stat["rmse"] is None or not np.isfinite(stat["rmse"]) or not np.isfinite(mean_ref) or mean_ref == 0:
            continue
        terms.append((stat["rmse"] / mean_ref) ** 2)
    if not terms:
        return np.nan
    return 100.0*scale_ratio * np.sqrt(sum(terms) / len(terms))


def format_table(stats, descriptions=None):
    header = "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    lines = [
        header.format(
            "Band", "Valid", "MSE", "RMSE", "MAE", "PSNR", "SSIM", "UIQI", "Bias", "Corr"
        )
    ]
    for row in stats:
        label = descriptions[row["band"] - 1] if descriptions and descriptions[row["band"] - 1] else row["band"]
        lines.append(
            header.format(
                label,
                row["valid_pixels"],
                f"{row['mse']:.6f}" if np.isfinite(row["mse"]) else "nan",
                f"{row['rmse']:.6f}" if np.isfinite(row["rmse"]) else "nan",
                f"{row['mae']:.6f}" if np.isfinite(row["mae"]) else "nan",
                f"{row['psnr']:.2f}" if np.isfinite(row["psnr"]) else "nan",
                f"{row['ssim']:.4f}" if np.isfinite(row["ssim"]) else "nan",
                f"{row['uiqi']:.4f}" if np.isfinite(row["uiqi"]) else "nan",
                f"{row['bias']:.6f}" if np.isfinite(row["bias"]) else "nan",
                f"{row['corr']:.3f}" if np.isfinite(row["corr"]) else "nan",
            )
        )
    return "\n".join(lines)


def save_csv(stats, path: Path, descriptions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("band,label,valid_pixels,mse,rmse,mae,psnr,ssim,uiqi,bias,corr\n")
        for row in stats:
            label = descriptions[row["band"] - 1] if descriptions and descriptions[row["band"] - 1] else ""
            f.write(
                f"{row['band']},{label},{row['valid_pixels']},{row['mse']},"
                f"{row['rmse']},{row['mae']},{row['psnr']},{row['ssim']},"
                f"{row['uiqi']},{row['bias']},{row['corr']}\n"
            )


def save_excel(stats, path: Path, descriptions=None):
    if Workbook is None:
        raise ImportError(
            "No se pudo exportar a Excel porque openpyxl no esta instalado. "
            "Instala con: pip install openpyxl"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"

    headers = [
        "band",
        "label",
        "valid_pixels",
        "mse",
        "rmse",
        "mae",
        "psnr",
        "ssim",
        "uiqi",
        "bias",
        "corr",
    ]
    ws.append(headers)

    for row in stats:
        label = descriptions[row["band"] - 1] if descriptions and descriptions[row["band"] - 1] else ""
        ws.append(
            [
                row["band"],
                label,
                row["valid_pixels"],
                row["mse"],
                row["rmse"],
                row["mae"],
                row["psnr"],
                row["ssim"],
                row["uiqi"],
                row["bias"],
                row["corr"],
            ]
        )

    wb.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Calcula métricas entre predicción y referencia.")
    parser.add_argument("--prediction", required=True, help="GeoTIFF superresuelto.")
    parser.add_argument("--reference", required=True, help="GeoTIFF referencia (dron).")
    parser.add_argument("--data-range", type=float, default=2.0, help="Rango dinámico esperado.")
    parser.add_argument("--ergas-scale", type=float, default=1.0, help="Relación de resoluciones para ERGAS.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Archivo CSV opcional.")
    parser.add_argument("--output-xlsx", type=Path, default=None, help="Archivo Excel (.xlsx) opcional.")
    return parser.parse_args()


def main():
    args = parse_args()
    pred, pred_desc = read_raster(Path(args.prediction))
    ref, ref_desc = read_raster(Path(args.reference))
    stats, ref_means = compute_band_metrics(pred, ref, data_range=args.data_range)
    descriptions = pred_desc if any(pred_desc) else ref_desc
    print(format_table(stats, descriptions))

    sam_mean, sam_median = spectral_angle_mapper(pred, ref)
    sid_mean = spectral_information_divergence(pred, ref)
    ergas_val = ergas(stats, ref_means, scale_ratio=args.ergas_scale)

    print("\n--- Métricas globales ---")
    print(f"SAM (media/mediana) [grados]: {sam_mean:.3f} / {sam_median:.3f}" if np.isfinite(sam_mean) else "SAM: sin datos válidos")
    print(f"SID (media): {sid_mean:.6f}" if np.isfinite(sid_mean) else "SID: sin datos válidos")
    print(f"ERGAS: {ergas_val:.3f}" if np.isfinite(ergas_val) else "ERGAS: sin datos válidos")

    if args.output_csv:
        save_csv(stats, args.output_csv, descriptions)
        print(f"Métricas guardadas en: {args.output_csv}")

    if args.output_xlsx:
        save_excel(stats, args.output_xlsx, descriptions)
        print(f"Métricas Excel guardadas en: {args.output_xlsx}")


if __name__ == "__main__":
    main()
