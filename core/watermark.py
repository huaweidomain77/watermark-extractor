"""
core/watermark.py
Isolate the red watermark by intersecting the red mask with MSER text regions.

Updated for Tesseract OCR:
- Filters out small noise blobs before cropping (connected-component analysis)
- Renders as clean black-on-white instead of preserving red coloring
  (Tesseract is trained on black/dark text, not colored text)
- Tighter bounding-box crop based on the FILTERED mask, not the raw mask
- Light dilation to thicken thin strokes for cleaner OCR
"""
import io
import cv2
import numpy as np
from PIL import Image as PILImage

from core.red_mask import _red_masks
from core.text_regions import _extract_text_regions_multi


MIN_WATERMARK_PIXELS = 100

# Connected components smaller than this (in pixels) are treated as
# noise and removed before cropping. Tune this if real characters
# start getting dropped (lower it) or noise still gets through
# (raise it).
MIN_COMPONENT_AREA = 15


def _remove_small_components(mask, min_area=MIN_COMPONENT_AREA):
    """
    Removes connected components smaller than min_area pixels.

    This strips out scattered noise speckles (isolated red pixels
    unrelated to the actual watermark text) that would otherwise
    widen the crop bounding box and confuse OCR.
    """
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=bool)

    # label 0 is background, skip it
    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:

            cleaned |= (labels == label)

    return cleaned


def isolate_watermark_image(image_bytes, dilate_size=1, debug=False):
    """
    Intersects red-mask (HSV strict) with MSER text-region mask. Keeps
    only pixels that are BOTH red AND text-shaped. Filters out small
    noise blobs, converts to clean black-on-white, crops tightly to
    the bounding box, and upscales.
    """
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)

    strict_mask, _ = _red_masks(arr)

    text_mask = _extract_text_regions_multi(arr, debug=debug) > 0
    combined = strict_mask & text_mask

    raw_pixel_count = int(combined.sum())

    if raw_pixel_count == 0:
        buf = io.BytesIO()
        PILImage.new("RGB", (10, 10), (255, 255, 255)).save(buf, format="PNG")
        buf.seek(0)
        return buf.read(), 0

    # ----------------------------------------------------------------
    # STEP 1: Remove small noise blobs before doing anything else.
    # ----------------------------------------------------------------

    cleaned = _remove_small_components(combined)

    # If cleaning wiped out everything (shouldn't normally happen,
    # but guard against it), fall back to the original combined mask
    # rather than returning a blank image.
    if cleaned.sum() == 0:
        cleaned = combined

    # ----------------------------------------------------------------
    # STEP 2: Light dilation to thicken thin strokes.
    #
    # Upscaled thin red lines often break into disconnected
    # fragments (visible in your debug image as split characters).
    # A small dilation merges those back into solid strokes.
    # ----------------------------------------------------------------

    kernel = np.ones((2, 2), np.uint8)

    dilated = cv2.dilate(
        cleaned.astype(np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)

    # ----------------------------------------------------------------
    # STEP 3: Render as clean BLACK text on WHITE background.
    #
    # Tesseract is trained on black/dark text on light backgrounds.
    # Keeping the original red coloring works against it.
    # ----------------------------------------------------------------

    out = np.full(arr.shape, 255, dtype="uint8")
    out[dilated] = [0, 0, 0]

    out_img = PILImage.fromarray(out)

    # ----------------------------------------------------------------
    # STEP 4: Tight bounding-box crop, based on the CLEANED mask
    # (not the noisy raw mask), so scattered speckles no longer
    # drag the crop wider than the actual text block.
    # ----------------------------------------------------------------

    ys, xs = np.where(dilated)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    padding = 15

    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(out_img.height - 1, y1 + padding)
    x1 = min(out_img.width - 1, x1 + padding)

    out_img = out_img.crop((x0, y0, x1, y1))

    # DEBUG: save isolated watermark before upscaling
    debug_path = "debug_isolated_watermark.png"
    out_img.save(debug_path)
    print(f"  🖼️ DEBUG isolated watermark saved: {debug_path}")

    # ----------------------------------------------------------------
    # STEP 5: Upscale for OCR. 4x instead of 3x since we're now
    # feeding clean black/white to Tesseract rather than a vision
    # model, and Tesseract benefits from larger character size.
    # ----------------------------------------------------------------

    out_img = out_img.resize(
        (out_img.width * 4, out_img.height * 4),
        PILImage.LANCZOS,
    )

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read(), raw_pixel_count