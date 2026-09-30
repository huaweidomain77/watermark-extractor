"""
core/validation.py

Standalone sanity-check validation for extracted watermark data.

This has ZERO dependency on OpenAI -- pure Python checks only.
Used by extractor.py's _is_good_extraction() as an additional
layer of validation beyond the basic structure checks.
"""


# Loose geographic bounding box for Pakistan, since that's where
# this project's watermarks are expected to originate (per the
# original OpenAI prompt's "lat/lng = positive numbers, Pakistan").
# Adjust these if survey sites fall outside this range.
PAKISTAN_LAT_MIN = 23.0
PAKISTAN_LAT_MAX = 37.5
PAKISTAN_LON_MIN = 60.0
PAKISTAN_LON_MAX = 78.0


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

    if not (PAKISTAN_LAT_MIN <= lat <= PAKISTAN_LAT_MAX):
        return False, f"latitude {lat} outside expected range"

    if not (PAKISTAN_LON_MIN <= lon <= PAKISTAN_LON_MAX):
        return False, f"longitude {lon} outside expected range"

    return True, None