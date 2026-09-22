"""
core/text_regions.py
MSER-based text region detection (0° and 90° union).
"""
import cv2
import numpy as np
from core.preprocess import _preprocess_for_mser


def _extract_text_regions(arr, debug=False):
    """
    Uses MSER to find character-shaped regions. Returns a uint8 mask
    (0 = not text, 255 = text-shaped). Works on shape, not color.
    """
    gray = _preprocess_for_mser(arr)
    mser = cv2.MSER_create()
    mser.setMinArea(15)
    mser.setMaxArea(4000)
    regions, _ = mser.detectRegions(gray)

    keep = np.zeros(gray.shape, dtype=np.uint8)
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        aspect = w / max(h, 1)
        if not (0.08 < aspect < 12):
            continue
        if w > 120 or h > 120:
            continue
        if w < 4 or h < 6:
            continue
        keep[y:y + h, x:x + w] = 255

    if debug:
        print(f"  [mser] {len(regions)} raw regions, "
              f"{int((keep > 0).sum())} text-shaped pixels")

    return keep


def _extract_text_regions_multi(arr, debug=False):
    """Run MSER at 0° and 90°, union the masks."""
    m1 = _extract_text_regions(arr, debug=debug)
    arr90 = np.rot90(arr, k=1).copy()
    m2 = np.rot90(_extract_text_regions(arr90, debug=False), k=-1)
    return np.maximum(m1, m2)