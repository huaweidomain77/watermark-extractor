"""
core/validation.py

Standalone sanity-check validation for extracted watermark data.

No geographic assumptions -- the actual location validation in
this system is the 300m distance check against the master file
(core/distance.py), not a hardcoded country/region box. This
function only catches obviously broken data (missing/non-numeric
coordinates, or the common OCR-failure sentinel of 0,0).
"""


def validate_extraction(data):
    """
    Extra sanity checks beyond basic structure/format validation.

    Returns (valid: bool, reason: str|None)
    """

    if not isinstance(data, dict):
        return False, "not a dict"

    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if latitude is None or longitude is None:
        return False, "missing coordinates"

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False, "coordinates not numeric"

    # Common OCR-failure sentinel -- reject exact 0,0 rather than
    # silently treating it as a real location.
    if lat == 0 and lon == 0:
        return False, "coordinates are zero (likely OCR failure)"

    return True, None