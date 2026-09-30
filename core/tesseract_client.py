"""
core/tesseract_client.py

Local OCR-based text extraction + parsing for the watermark format.

Replaces core/openai_client.py's model-calling role inside
extractor.py. No API key, no network calls, no per-image cost --
purely local Tesseract OCR + regex parsing.

Returns (data, reason) tuples to match the shape extractor.py
already expects, so the rest of the pipeline (rotation retries,
validation, fallback) needs minimal changes.
"""

import io
import re

import pytesseract
from PIL import Image

from utils.helpers import (
    clean_coord,
    force_string,
    clean_side_id,
    truncate_to_8_decimals,
)


def _rotate_image_bytes(image_bytes, degrees):

    if degrees == 0:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes))

    rotated = img.rotate(
        -degrees,
        expand=True,
        fillcolor=(255, 255, 255),
    )

    buf = io.BytesIO()
    rotated.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _run_ocr(image_bytes):
    """
    Runs Tesseract on the image and returns raw multi-line text.

    --psm 6: treat the image as a single uniform block of text
    (better for a tightly cropped watermark than the default
    "assume a full page" mode).

    Whitelisting characters cuts down on noise from stray marks
    or leftover artifacts being misread as symbols.
    """

    img = Image.open(io.BytesIO(image_bytes))

    config = (
        "--psm 6 "
        "-c tessedit_char_whitelist="
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ:-. "
    )

    text = pytesseract.image_to_string(img, config=config)

    return text


def _find_coordinates(text):
    """
    Finds latitude/longitude-looking decimal numbers in the text.

    Coordinates in this project are written with many decimal
    places (e.g. 31.81509078), so we look for numbers with at
    least 4 digits after the decimal point -- this avoids
    matching unrelated short decimals or OCR noise.
    """

    candidates = re.findall(r"-?\d{1,3}\.\d{4,8}", text)

    lat = None
    lon = None

    for value in candidates:

        try:
            number = float(value)
        except ValueError:
            continue

        if lat is None and abs(number) <= 90:
            lat = value
            continue

        if lon is None and abs(number) <= 180:
            lon = value

    return lat, lon


def _find_side_id(text):
    """
    Looks for a Side ID matching known formats:
    - LETTERS-DIGITS  (e.g. CII-2842)
    - 4 digit numeric (e.g. 2842)
    """

    match = re.search(r"\b[A-Z]{1,6}-\d{3,6}\b", text)

    if match:
        return match.group(0)

    match = re.search(r"\b\d{4}\b", text)

    if match:
        return match.group(0)

    return None


def _find_mp_number(text):
    """
    Looks for the MP number format: YYYYMMDD-NNNN
    """

    match = re.search(r"\b\d{8}-\d{3,6}\b", text)

    if match:
        return match.group(0)

    return None


def extract_from_image(image_bytes, degrees=0):
    """
    Main entry point: rotates the image if needed, runs OCR, parses
    out the structured watermark fields, and applies the same
    cleaning steps the old OpenAI pipeline used (clean_coord,
    clean_side_id, truncate_to_8_decimals) so downstream validation
    in extractor.py sees data in the same shape as before.

    Returns (data, reason) -- same shape extractor.py already
    expects from _call_model_once.
    """

    try:

        rotated_bytes = _rotate_image_bytes(
            image_bytes,
            degrees,
        )

        raw_text = _run_ocr(rotated_bytes)

        print(f"      [tesseract raw] {raw_text!r}")

        if not raw_text or not raw_text.strip():
            return None, "No text detected"

        latitude, longitude = _find_coordinates(raw_text)
        side_id = _find_side_id(raw_text)
        mp_number = _find_mp_number(raw_text)

        data = {
            "side_id": side_id,
            "side_id_clear": side_id is not None,
            "latitude": latitude,
            "latitude_clear": latitude is not None,
            "longitude": longitude,
            "longitude_clear": longitude is not None,
            "mp_number": mp_number,
        }

        # Same cleaning steps the OpenAI pipeline applied, so
        # extractor.py's validation sees consistent formatting.
        data["latitude"] = clean_coord(force_string(data.get("latitude")))
        data["longitude"] = clean_coord(force_string(data.get("longitude")))
        data["side_id"] = clean_side_id(force_string(data.get("side_id")))
        data["mp_number"] = force_string(data.get("mp_number"))

        data["latitude"] = truncate_to_8_decimals(data.get("latitude"))
        data["longitude"] = truncate_to_8_decimals(data.get("longitude"))

        return data, None

    except Exception as e:

        return None, str(e)