"""
core/red_mask.py
HSV-based red watermark mask detection.
"""
import numpy as np


def _red_masks(arr):
    """
    Returns (strict_mask, loose_mask) as boolean arrays, using HSV so
    vivid red watermark text is separated from muted red surfaces
    (wood, pink paper) by SATURATION and VALUE, not just RGB deltas.
    """
    rgb = arr.astype(np.float32) / 255.0
    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    v = maxc
    delta = maxc - minc

    s = np.where(maxc > 1e-6, delta / np.maximum(maxc, 1e-6), 0.0)

    r_, g_, b_ = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h = np.zeros_like(maxc)
    nz = delta > 1e-6
    mask_r = nz & (maxc == r_)
    mask_g = nz & (maxc == g_) & ~mask_r
    mask_b = nz & (maxc == b_) & ~mask_r & ~mask_g

    h[mask_r] = (60 * ((g_ - b_) / np.maximum(delta, 1e-6)) % 360)[mask_r]
    h[mask_g] = (60 * ((b_ - r_) / np.maximum(delta, 1e-6)) + 120)[mask_g]
    h[mask_b] = (60 * ((r_ - g_) / np.maximum(delta, 1e-6)) + 240)[mask_b]
    h = h % 360

    hue_is_red = (h < 25) | (h > 335)

    strict = hue_is_red & (s > 0.55) & (v > 0.55)
    loose = hue_is_red & (s > 0.25) & (v > 0.40)

    return strict, loose