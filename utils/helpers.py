"""
utils/helpers.py
Shared helper functions used across the pipeline.
"""
import re
import io
import base64
import numpy as np
from PIL import Image as PILImage


# ============================================================
# STRING / NUMBER CLEANING
# ============================================================

def clean_coord(value):
    """Strip stray +/- signs — Pakistan coords are always positive."""
    if value is None:
        return None
    return str(value).replace("+", "").replace("-", "").strip()


def force_string(value):
    """Ensure numeric fields stay as exact strings — no precision/zero loss."""
    if value is None:
        return None
    if isinstance(value, float):
        formatted = f"{value:.8f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def clean_side_id(value):
    """Normalize side_id spacing WITHOUT touching a letter prefix or hyphen."""
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"\s*-\s*", "-", s)
    return s


def truncate_to_8_decimals(value):
    """Truncate (never round) a coordinate string to 8 digits after the decimal."""
    if value is None or value == "":
        return value
    s = str(value).strip()
    if "." not in s:
        return s
    whole, frac = s.split(".", 1)
    frac = frac[:8]
    return f"{whole}.{frac}" if frac else whole


def get_media_type(filename):
    ext = filename.lower().split(".")[-1]
    return {"png": "image/png", "jpg": "image/jpeg",
            "jpeg": "image/jpeg"}.get(ext, "image/jpeg")


# ============================================================
# IMAGE I/O
# ============================================================

def _cap_longest_side(png_bytes, max_side=1600):
    """Resize PNG so its longest side is at most max_side pixels."""
    try:
        im = PILImage.open(io.BytesIO(png_bytes))
        w, h = im.size
        if max(w, h) <= max_side:
            return png_bytes
        scale = max_side / float(max(w, h))
        im = im.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return png_bytes


def _rotate_isolated_png(png_bytes, degrees):
    """Rotate an isolated watermark PNG by the given degrees."""
    if degrees == 0:
        return png_bytes
    img = PILImage.open(io.BytesIO(png_bytes))
    rotated = img.rotate(degrees, expand=True, fillcolor=(255, 255, 255),
                         resample=PILImage.NEAREST)
    buf = io.BytesIO()
    rotated.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()