"""
core/watermark.py
Isolate the red watermark by intersecting the red mask with MSER text regions.
"""
import io
import numpy as np
from PIL import Image as PILImage

from core.red_mask import _red_masks
from core.text_regions import _extract_text_regions_multi


MIN_WATERMARK_PIXELS = 100


def isolate_watermark_image(image_bytes, dilate_size=1, debug=False,
                            save_debug=False, debug_name=None):
    """
    Intersects red-mask (HSV strict) with MSER text-region mask. Keeps
    only pixels that are BOTH red AND text-shaped. Then crops to the
    bounding box of the mask and 3x upscales.
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

    # Keep original color, darken watermark pixels for contrast
    out = np.full(arr.shape, 255, dtype="uint8")
    out[combined] = arr[combined]
    out[combined, 1] = (arr[combined, 1] * 0.35).astype(np.uint8)
    out[combined, 2] = (arr[combined, 2] * 0.35).astype(np.uint8)

    out_img = PILImage.fromarray(out)

    # Bounding-box crop around detected pixels
    ys, xs = np.where(combined)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - 10)
    x0 = max(0, x0 - 10)
    y1 = min(out_img.height - 1, y1 + 10)
    x1 = min(out_img.width - 1, x1 + 10)
    out_img = out_img.crop((x0, y0, x1, y1))

    # DEBUG: save isolated watermark before upscaling.
    # Off by default, and given a unique name per call, because a
    # fixed shared filename collides when multiple images are
    # processed concurrently (each thread would overwrite the same
    # file mid-write). Pass save_debug=True + a distinct debug_name
    # for manual single-image debugging only.
    if save_debug:
        try:
            name = debug_name or f"debug_isolated_watermark_{id(image_bytes)}.png"
            out_img.save(name)
            print(f"  🖼️ DEBUG isolated watermark saved: {name}")
        except Exception as e:
            print(f"  ⚠️ Could not save debug image: {e}")

    # 3x upscale
    out_img = out_img.resize((out_img.width * 3, out_img.height * 3),
                             PILImage.LANCZOS)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read(), raw_pixel_count