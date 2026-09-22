"""
core/corners.py
Corner / edge-strip detection to find the red watermark region.
"""
import io
import numpy as np
from PIL import Image as PILImage

from core.red_mask import _red_masks
from core.text_regions import _extract_text_regions


def _get_crop_regions(img, crop_fraction):
    """Return dict of named crop regions (corners + edge strips)."""
    w, h = img.size
    cw = int(w * crop_fraction)
    ch = int(h * crop_fraction)
    return {
        "top_left":     img.crop((0,    0,    cw,  ch)),
        "top_right":    img.crop((w-cw, 0,    w,   ch)),
        "bottom_left":  img.crop((0,    h-ch, cw,  h)),
        "bottom_right": img.crop((w-cw, h-ch, w,   h)),
        "left_strip":   img.crop((0,    0,    cw,  h)),
        "right_strip":  img.crop((w-cw, 0,    w,   h)),
        "top_strip":    img.crop((0,    0,    w,   ch)),
        "bottom_strip": img.crop((0,    h-ch, w,   h)),
    }


def _red_orange_score(crop_img):
    """Density-weighted red-and-text score."""
    arr = np.array(crop_img)
    strict_mask, _ = _red_masks(arr)
    text_mask = _extract_text_regions(arr) > 0
    combined = strict_mask & text_mask
    n = int(combined.sum())
    if n == 0:
        return 0
    ys, xs = np.where(combined)
    h_spread = int(ys.max() - ys.min()) + 1
    w_spread = int(xs.max() - xs.min()) + 1
    return int(n * (n / (h_spread * w_spread)))


def find_watermark_corner(image_bytes, crop_fraction=0.25):
    """
    Tries all 4 corners + 4 edge strips, returns the one with the most
    red-and-text-shaped pixels.
    """
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    corners = _get_crop_regions(img, crop_fraction)
    best_corner = max(corners, key=lambda k: _red_orange_score(corners[k]))
    best_crop = corners[best_corner]

    buffer = io.BytesIO()
    best_crop.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read(), best_corner


def get_corner_score(image_bytes, corner, crop_fraction=0.25):
    """Score a specific corner region."""
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    corners = _get_crop_regions(img, crop_fraction)
    return _red_orange_score(corners[corner])