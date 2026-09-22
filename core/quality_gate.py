"""
core/quality_gate.py
Reject low-confidence extractions before they enter the pipeline.
"""
import re


PK_LAT_RANGE = (23.0, 38.0)
PK_LON_RANGE = (60.0, 78.0)

SIDE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-\s]{2,80}$")

SIDE_ID_BLOCKLIST = {"guarded", "unguarded", "un guarded", "ok", "nok",
                     "none", "null", "n/a", "na", "yes", "no"}


def _looks_like_valid_side_id(value):
    if not value:
        return False
    s = str(value).strip()
    if not SIDE_ID_PATTERN.match(s):
        return False
    if s.lower() in SIDE_ID_BLOCKLIST:
        return False
    if not any(ch.isdigit() for ch in s):
        return False
    return True


def _looks_like_valid_coord(value, low, high):
    if value in (None, ""):
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return low <= f <= high


def apply_quality_gate(data, master_ids=None):
    """Apply confidence + range checks. Mutates and returns `data`."""
    side_id_ok = (data.get("side_id_clear") is True
                  and _looks_like_valid_side_id(data.get("side_id")))
    if side_id_ok and master_ids is not None:
        side_id_ok = str(data.get("side_id")).strip() in master_ids
    data["side_id"] = data.get("side_id") if side_id_ok else "Too blurry to extract"

    lat_ok = (data.get("latitude_clear") is True
              and _looks_like_valid_coord(data.get("latitude"), *PK_LAT_RANGE))
    data["latitude"] = data.get("latitude") if lat_ok else ""

    lon_ok = (data.get("longitude_clear") is True
              and _looks_like_valid_coord(data.get("longitude"), *PK_LON_RANGE))
    data["longitude"] = data.get("longitude") if lon_ok else ""

    return data