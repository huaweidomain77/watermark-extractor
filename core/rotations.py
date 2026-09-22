"""
core/rotations.py

Local watermark orientation detection.

Important:
The watermark contains multiple horizontal text lines, so we must NOT
determine orientation from the overall bounding-box aspect ratio.
"""

import io
import cv2
import numpy as np
from PIL import Image as PILImage

from core.red_mask import _red_masks
from core.text_regions import _extract_text_regions


# ============================================================
# ORIENTATION SCORING
# ============================================================

def _directional_line_score(mask, direction):
    """
    Measure how strongly the text pixels form lines in a direction.

    Horizontal text should produce long connected components when a
    horizontal morphological kernel is used.

    Vertical text should produce long connected components when a
    vertical kernel is used.
    """
    mask = (mask > 0).astype(np.uint8) * 255

    h, w = mask.shape

    # Kernel size scales with image dimensions.
    # Keep it reasonably small so we don't accidentally merge
    # unrelated regions.
    k = max(15, min(51, int(min(h, w) * 0.025)))

    # Kernel dimensions must be odd.
    if k % 2 == 0:
        k += 1

    if direction == "horizontal":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (k, 3),
        )
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, k),
        )

    # Dilate characters toward the expected text-line direction.
    dilated = cv2.dilate(mask, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated,
        connectivity=8,
    )

    if num_labels <= 1:
        return 0.0

    original_pixels = max(1, int(mask.sum() / 255))

    score = 0.0

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Ignore tiny noise.
        if area < 100:
            continue

        short_side = max(1, min(width, height))
        long_side = max(width, height)

        elongation = long_side / short_side

        # We want elongated line-like components.
        if elongation < 1.5:
            continue

        # Weight by area and elongation.
        score += area * elongation

    return score / original_pixels


def _estimate_orientation_from_mask(mask):
    """
    Estimate whether text is horizontal or vertical.

    Returns:
        0   -> horizontal
        90  -> vertical
    """

    horizontal_score = _directional_line_score(
        mask,
        "horizontal",
    )

    vertical_score = _directional_line_score(
        mask,
        "vertical",
    )

    print(
        f"  📐 orientation scores: "
        f"horizontal={horizontal_score:.3f}, "
        f"vertical={vertical_score:.3f}"
    )

    if horizontal_score <= 0 and vertical_score <= 0:
        print("  ⚠️ Orientation ambiguous — defaulting to 0°")
        return 0

    # Require a meaningful margin before declaring vertical.
    #
    # This is important:
    # ambiguous/noisy cases should NOT cause expensive 90° API calls.
    if vertical_score > horizontal_score * 1.25:
        return 90

    return 0


# ============================================================
# ISOLATED WATERMARK
# ============================================================

def estimate_watermark_angle(isolated_png_bytes):
    """
    Estimate orientation of an isolated watermark.

    IMPORTANT:
    We intentionally do NOT use minAreaRect() on the complete
    watermark bounding box because a multi-line horizontal watermark
    is naturally taller than it is wide.
    """

    img = PILImage.open(
        io.BytesIO(isolated_png_bytes)
    ).convert("L")

    arr = np.array(img)

    # Dark text on white background.
    mask = (arr < 128).astype(np.uint8) * 255

    if int(mask.sum() / 255) < 20:
        return 0

    return _estimate_orientation_from_mask(mask)


# ============================================================
# RAW COLOR CROP
# ============================================================

def estimate_raw_crop_angle(crop_bytes):
    """
    Estimate orientation from red watermark pixels in a raw crop.
    """

    img = PILImage.open(
        io.BytesIO(crop_bytes)
    ).convert("RGB")

    arr = np.array(img)

    strict_mask, _ = _red_masks(arr)

    # Keep only red pixels that also look like text.
    text_mask = _extract_text_regions(arr) > 0

    combined = (
        strict_mask & text_mask
    ).astype(np.uint8) * 255

    if int(combined.sum() / 255) < 50:
        print("  ⚠️ Raw orientation mask too small — defaulting to 0°")
        return 0

    return _estimate_orientation_from_mask(combined)


# ============================================================
# ROTATION CANDIDATES
# ============================================================

def _rotation_candidates(coarse_angle):
    """
    Return candidates in priority order.

    The detected orientation is ALWAYS tried first.

    We keep 180° as the next candidate because a watermark can be
    upside-down while still appearing horizontally aligned.
    """

    coarse_angle = int(coarse_angle) % 360

    order = {
        0:   [0, 180, 90, 270],
        90:  [90, 270, 0, 180],
        180: [180, 0, 90, 270],
        270: [270, 90, 0, 180],
    }

    return order.get(
        coarse_angle,
        [0, 180, 90, 270],
    )