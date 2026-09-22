"""
core/extractor.py

Main watermark extraction orchestrator.

Important behavior:

- Every image is processed independently.
- Same Side ID can appear in multiple images.
- Coordinates are never copied from another image.
- Filename is only a hint.
- Normal extraction and fallback extraction use the same validation.
- Suspicious/repeated observations can trigger an independent
  verification pass.
"""

import io
import re

import cv2
import numpy as np

from PIL import Image as PILImage
from PIL import Image, ImageEnhance, ImageFilter

from utils.helpers import (
    _cap_longest_side,
    truncate_to_8_decimals,
)

from core.openai_client import (
    _call_model_once,
    RAW_CROP_PROMPT,
    validate_extraction,
)

from core.corners import (
    find_watermark_corner,
    get_corner_score,
)

from core.watermark import (
    isolate_watermark_image,
    MIN_WATERMARK_PIXELS,
)

from core.rotations import (
    estimate_watermark_angle,
    estimate_raw_crop_angle,
    _rotation_candidates,
)


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def _enhance_for_ocr(image_bytes):
    """
    Create a high-contrast grayscale version for OCR.
    """

    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")

    img = ImageEnhance.Contrast(img).enhance(2.5)

    img = img.filter(
        ImageFilter.SHARPEN
    )

    img = img.filter(
        ImageFilter.SHARPEN
    )

    min_width = 1600

    if img.width < min_width:

        ratio = min_width / img.width

        img = img.resize(
            (
                int(img.width * ratio),
                int(img.height * ratio),
            ),
            Image.LANCZOS,
        )

    buf = io.BytesIO()

    img.save(
        buf,
        format="PNG",
    )

    return buf.getvalue()


# ============================================================
# SHARPNESS
# ============================================================

def _estimate_sharpness(isolated_png_bytes):

    img = PILImage.open(
        io.BytesIO(isolated_png_bytes)
    ).convert("L")

    arr = np.array(img)

    return float(
        cv2.Laplacian(
            arr,
            cv2.CV_64F,
        ).var()
    )


# ============================================================
# SIDE ID VALIDATION
# ============================================================

def _side_id_is_valid(side_id):

    if side_id is None:
        return False

    value = str(side_id).strip().upper()

    if not value:
        return False

    # Numeric IDs must be exactly four digits.
    if value.isdigit():
        return len(value) == 4

    # Known alphanumeric formats.
    if re.fullmatch(
        r"[A-Z]+-\d{4}",
        value,
    ):
        return True

    # Generic future format.
    if re.fullmatch(
        r"[A-Z0-9]+(?:-[A-Z0-9]+)+",
        value,
    ):
        return True

    return False


# ============================================================
# COORDINATE VALIDATION
# ============================================================

def _coordinate_is_valid(value, minimum, maximum):

    if value is None:
        return False

    try:

        number = float(value)

        return minimum <= number <= maximum

    except Exception:
        return False


def _coordinates_are_valid(data):

    if not data:
        return False

    lat = data.get("latitude")
    lon = data.get("longitude")

    return (
        _coordinate_is_valid(
            lat,
            -90,
            90,
        )
        and
        _coordinate_is_valid(
            lon,
            -180,
            180,
        )
    )


# ============================================================
# EXTRACTION QUALITY CHECK
# ============================================================

def _is_good_extraction(data):

    """
    Final local validation.

    This function is intentionally strict.
    """

    if not isinstance(data, dict):
        return False

    side_id = data.get("side_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    # Required fields.
    if not side_id:
        return False

    if not latitude:
        return False

    if not longitude:
        return False

    if str(side_id).strip() == "Too blurry to extract":
        return False

    # Side ID structure.
    if not _side_id_is_valid(side_id):
        print(
            f"    ⚠️ Invalid Side ID rejected: "
            f"{side_id!r}"
        )
        return False

    # Coordinate structure.
    if not _coordinates_are_valid(data):

        print(
            f"    ⚠️ Invalid coordinates rejected: "
            f"{latitude!r}, {longitude!r}"
        )

        return False

    # Respect model uncertainty flags.
    if data.get("side_id_clear") is False:
        print(
            "    ⚠️ Side ID marked unclear"
        )
        return False

    if data.get("latitude_clear") is False:
        print(
            "    ⚠️ Latitude marked unclear"
        )
        return False

    if data.get("longitude_clear") is False:
        print(
            "    ⚠️ Longitude marked unclear"
        )
        return False

    # Run client-side validation too.
    valid, reason = validate_extraction(data)

    if not valid:

        print(
            f"    ⚠️ Validation rejected: "
            f"{reason}"
        )

        return False

    return True


# ============================================================
# OBSERVATION COMPARISON
# ============================================================

def _same_coordinates(a, b, tolerance=0.00000001):

    """
    Compare two extracted coordinate pairs.

    Very small tolerance because your coordinates contain
    many decimal places.
    """

    try:

        a_lat = float(a.get("latitude"))
        a_lon = float(a.get("longitude"))

        b_lat = float(b.get("latitude"))
        b_lon = float(b.get("longitude"))

        return (
            abs(a_lat - b_lat) <= tolerance
            and
            abs(a_lon - b_lon) <= tolerance
        )

    except Exception:
        return False


def _same_side_id(a, b):

    try:

        return (
            str(a.get("side_id"))
            .strip()
            .upper()
            ==
            str(b.get("side_id"))
            .strip()
            .upper()
        )

    except Exception:

        return False


def _looks_suspicious_against_previous(
    data,
    previous_results,
):
    """
    Detect the exact problem visible in your screenshot:

        image 1:
        1683 -> coordinates A

        image 2:
        1683 -> coordinates A

    Repeated Side IDs are NOT suspicious by themselves.

    Repeated Side ID + EXACT SAME coordinates is what triggers
    another independent verification pass.
    """

    if not data:
        return False

    if not previous_results:
        return False

    for previous in previous_results:

        if not isinstance(previous, dict):
            continue

        if _same_side_id(data, previous):

            if _same_coordinates(
                data,
                previous,
            ):

                return True

    return False


# ============================================================
# ROTATION EXTRACTION
# ============================================================

def _try_rotations(
    isolated_png_bytes,
    filename,
    coarse_angle,
):
    """
    Try several rotations.

    Every result goes through strict validation.
    """

    candidates = _rotation_candidates(
        coarse_angle
    )

    for deg in candidates:

        print(
            f"   🔄 Trying rotation={deg}°"
        )

        data, reason = _call_model_once(
            isolated_png_bytes,
            filename,
            deg,
            max_tokens=4000,
        )

        if data is None:

            print(
                f"   ⚠️ Rotation {deg} rejected: "
                f"{reason}"
            )

            continue

        # IMPORTANT:
        # Never trust fallback/model output blindly.
        if not _is_good_extraction(data):

            print(
                f"   ⚠️ Rotation {deg} failed "
                f"local validation"
            )

            continue

        print(
            f"   ✅ Rotation {deg} accepted"
        )

        return data

    return None


# ============================================================
# INDEPENDENT VERIFICATION
# ============================================================

def _verify_extraction(
    isolated_png_bytes,
    filename,
    first_result,
):
    """
    Make an independent second vision call.

    Used when the first result is suspicious.
    """

    verification_prompt = r"""
VERIFY THE WATERMARK AGAIN.

This is a second independent reading.

Do NOT trust the previous extraction.

Read the CURRENT IMAGE again from the actual red/orange
watermark.

The fixed line structure is:

LINE 1 = datetime
LINE 2 = region/station code — IGNORE
LINE 3 = ACTUAL SIDE ID — USE THIS
LINE 4 = latitude — USE THIS
LINE 5 = longitude — USE THIS
LINE 6 = MP number — USE THIS

The Side ID MUST come from line 3.

Do not use the filename.

Do not use MP number as Side ID.

Read numeric Side IDs digit by digit.

For example:

6044

is 6044, not 0744.

Read every latitude/longitude digit exactly.

Return JSON ONLY:

{
  "side_id": string or null,
  "side_id_clear": boolean,
  "latitude": string or null,
  "latitude_clear": boolean,
  "longitude": string or null,
  "longitude_clear": boolean,
  "mp_number": string or null
}
"""

    print(
        "   🔎 Running independent verification..."
    )

    verified, reason = _call_model_once(
        isolated_png_bytes,
        filename,
        0,
        max_tokens=4000,
        prompt_override=verification_prompt,
    )

    if verified is None:

        print(
            f"   ⚠️ Verification failed: {reason}"
        )

        return None

    if not _is_good_extraction(verified):

        print(
            "   ⚠️ Verification result failed "
            "local validation"
        )

        return None

    # If verification gives us a valid independent reading,
    # use it instead of blindly trusting the first result.
    print(
        "   ✅ Independent verification accepted:"
    )

    print(
        f"      Side ID: {verified.get('side_id')}"
    )

    print(
        f"      Lat:     {verified.get('latitude')}"
    )

    print(
        f"      Lon:     {verified.get('longitude')}"
    )

    return verified


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_watermark(
    image_bytes,
    filename="uploaded.jpg",
    master_ids=None,
    previous_results=None,
):
    """
    Extract watermark information from ONE image.

    Parameters
    ----------
    image_bytes:
        Current image bytes.

    filename:
        Current image filename.

    master_ids:
        Optional set/list of known Side IDs.

    previous_results:
        Previously accepted observations from this batch.

        Used ONLY to detect suspicious exact coordinate reuse.

        Repeated Side IDs are allowed.
    """

    print()
    print(
        f"=================================================="
    )
    print(
        f"📷 Processing: {filename}"
    )
    print(
        f"=================================================="
    )

    # --------------------------------------------------------
    # STAGE 1: WATERMARK CORNER
    # --------------------------------------------------------

    try:

        _, corner = find_watermark_corner(
            image_bytes
        )

        score = get_corner_score(
            image_bytes,
            corner,
        )

        print(
            f"📍 Watermark at: "
            f"{corner} "
            f"(score={score})"
        )

    except Exception as e:

        print(
            f"⚠️ Corner detection failed: {e}"
        )

        corner = None

    # --------------------------------------------------------
    # STAGE 1A: ISOLATE WATERMARK
    # --------------------------------------------------------

    isolated_png_bytes = None

    if corner is not None:

        try:

            isolated_png_bytes, raw_pixel_count = (
                isolate_watermark_image(
                    image_bytes,
                )
            )

            print(
                f"   isolated pixel count={raw_pixel_count}"
            )

            if raw_pixel_count < MIN_WATERMARK_PIXELS:

                print(
                    f"⚠️ Too few watermark pixels "
                    f"({raw_pixel_count}) — "
                    f"treating isolation as failed"
                )

                isolated_png_bytes = None

        except Exception as e:

            print(
                f"⚠️ Watermark isolation failed: {e}"
            )

    # --------------------------------------------------------
    # STAGE 1B: EXTRACT
    # --------------------------------------------------------

    if isolated_png_bytes:

        try:

            sharpness = _estimate_sharpness(
                isolated_png_bytes
            )

            print(
                f"🔎 sharpness={sharpness:.2f}"
            )

        except Exception:

            pass

        try:

            coarse_angle = estimate_watermark_angle(
                isolated_png_bytes
            )

        except Exception:

            coarse_angle = 0

        print(
            f"   coarse angle={coarse_angle}°"
        )

        result = _try_rotations(
            isolated_png_bytes,
            filename,
            coarse_angle,
        )

        if result is not None:

            # Check for suspicious coordinate reuse.
            if _looks_suspicious_against_previous(
                result,
                previous_results,
            ):

                print(
                    "   ⚠️ Same Side ID + same coordinates "
                    "found in a previous image."
                )

                verified = _verify_extraction(
                    isolated_png_bytes,
                    filename,
                    result,
                )

                if verified is not None:

                    result = verified

            if _is_good_extraction(result):

                print(
                    f"✅ FINAL extraction: "
                    f"{result}"
                )

                return result

    # --------------------------------------------------------
    # STAGE 2: RAW CROP FALLBACK
    # --------------------------------------------------------

    print()
    print(
        "🔁 Starting raw-crop fallback..."
    )

    fallback_attempts = []

    try:

        fallback_angle = estimate_raw_crop_angle(
            image_bytes
        )

    except Exception:

        fallback_angle = 0

    fallback_attempts.append(
        (
            image_bytes,
            fallback_angle,
        )
    )

    # Also try original image at zero rotation.
    if fallback_angle != 0:

        fallback_attempts.append(
            (
                image_bytes,
                0,
            )
        )

    for raw_bytes, raw_angle in fallback_attempts:

        print(
            f"   fallback rotation={raw_angle}°"
        )

        candidates = _rotation_candidates(
            raw_angle
        )

        for deg in candidates:

            data, reason = _call_model_once(
                raw_bytes,
                filename,
                deg,
                max_tokens=4000,
                prompt_override=RAW_CROP_PROMPT,
            )

            if data is None:

                print(
                    f"   ⚠️ fallback rotation "
                    f"{deg} rejected: {reason}"
                )

                continue

            # CRITICAL:
            # Fallback must use EXACTLY the same validation.
            if not _is_good_extraction(data):

                print(
                    f"   ⚠️ fallback rotation "
                    f"{deg} produced invalid data"
                )

                continue

            # Check suspicious coordinate reuse.
            if _looks_suspicious_against_previous(
                data,
                previous_results,
            ):

                print(
                    "   ⚠️ Fallback result has the "
                    "same Side ID + same coordinates "
                    "as a previous image."
                )

                verified = _verify_extraction(
                    raw_bytes,
                    filename,
                    data,
                )

                if verified is not None:

                    data = verified

            if not _is_good_extraction(data):

                continue

            print(
                f"   ✅ fallback accepted: "
                f"{data}"
            )

            return data

    # --------------------------------------------------------
    # NOTHING VALID FOUND
    # --------------------------------------------------------

    print(
        f"❌ Could not obtain a valid extraction "
        f"for {filename}"
    )

    return {
        "side_id": "Too blurry to extract",
        "latitude": None,
        "longitude": None,
        "mp_number": None,
    }