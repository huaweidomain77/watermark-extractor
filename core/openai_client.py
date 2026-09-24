"""
core/openai_client.py

Strict vision extraction for GPS watermark images.

Watermark structure:

1. datetime
2. region/station code      <- IGNORE
3. ACTUAL SIDE ID           <- USE THIS
4. lat=<latitude>           <- USE THIS
5. lng=<longitude>          <- USE THIS
6. MP-<number>              <- USE THIS

Important:
- Every image is processed independently.
- Same Side ID across multiple images is VALID.
- Coordinates are always taken from the current image.
- Filename is only a verification hint, never the source of truth.
"""

import os
import re
import json
import base64

from openai import OpenAI

from utils.helpers import (
    _cap_longest_side,
    _rotate_isolated_png,
    clean_coord,
    force_string,
    clean_side_id,
    truncate_to_8_decimals,
)
from core.quality_gate import apply_quality_gate


# ============================================================
# CONFIG
# ============================================================

API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL_NAME = "gpt-5.6-luna"

# 2000 was sometimes ending with finish_reason='length'.
MAX_COMPLETION_TOKENS = 4000

client = OpenAI(api_key=API_KEY)


# ============================================================
# PROMPTS
# ============================================================

PROMPT = r"""
You are extracting a GPS camera watermark.

THIS IS A STRICT FIXED-LAYOUT EXTRACTION TASK.

Read ONLY the red/orange watermark text.

The watermark has this exact logical order:

LINE 1 = datetime
LINE 2 = region/station/site code
LINE 3 = ACTUAL SIDE ID
LINE 4 = latitude
LINE 5 = longitude
LINE 6 = MP number

CRITICAL RULES:

1. LINE 2 IS NOT THE SIDE ID.
2. NEVER return the region/station code as side_id.
3. side_id MUST come ONLY from LINE 3.
4. latitude MUST come ONLY from LINE 4.
5. longitude MUST come ONLY from LINE 5.
6. mp_number MUST come ONLY from LINE 6.
7. Never copy coordinates from another image.
8. Never infer coordinates from the filename.
9. Never infer Side ID from the MP number.
10. Never invent or guess missing digits.

Examples of valid Side IDs:

1683
6044
S-3802
CII-2842

Examples of things that are NOT Side IDs:

S2.LKN.FUEL01
LKN-FUEL01
KH
LKN
MP-20260815-0508

IMPORTANT FOR NUMBERS:

Read the Side ID digit-by-digit from LEFT TO RIGHT.

For example:

6044

must be:

6044

NOT:

0744
0644
604
044

For latitude and longitude, preserve every visible digit.

Example (this is ONLY to show the format — these are fake numbers,
never copy them, always read the real numbers from the image):

lat=12.34567890
lng=98.76543210

must produce:

latitude = "12.34567890"
longitude = "98.76543210"

Do not round them.

If a digit is genuinely unreadable, return null for that field
instead of guessing.

Ignore:
- printed forms
- handwritten annotations
- FSR numbers
- TT numbers
- other black/blue text
- Guarded/Unguarded text
- fuel text
- any other non-watermark fields

Return ONLY valid JSON.

Schema:

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


RAW_CROP_PROMPT = r"""
Extract ONLY the red/orange GPS watermark from this image.

The watermark has this exact order:

LINE 1: datetime
LINE 2: region/station code
LINE 3: ACTUAL SIDE ID
LINE 4: lat=<latitude>
LINE 5: lng=<longitude>
LINE 6: MP-<number>

VERY IMPORTANT:

LINE 2 IS NOT THE SIDE ID.

The ONLY source for side_id is LINE 3.

The ONLY source for latitude is LINE 4.

The ONLY source for longitude is LINE 5.

The ONLY source for mp_number is LINE 6.

Valid Side ID examples:

1683
6044
S-3802
CII-2842

Invalid Side ID examples:

S2.LKN.FUEL01
LKN-FUEL01
KH
LKN

Read numeric Side IDs digit-by-digit.

If the visible Side ID is:

6044

return:

6044

Do NOT change it to:

0744
0644
604
044

Do not use the filename as the Side ID.

Do not use MP number as Side ID.

Do not copy coordinates from another image.

Do not guess unclear digits.

Return ONLY valid JSON in this exact schema:

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


# ============================================================
# JSON PARSER
# ============================================================

def _extract_json_object(text):
    """
    Robustly extract one JSON object.

    We don't rely on a fragile regex because model output can
    contain whitespace, code fences, or small extra text.
    """

    if not text:
        return None

    text = str(text).strip()

    # Remove markdown fences if they somehow appear.
    text = text.replace("```json", "")
    text = text.replace("```JSON", "")
    text = text.replace("```", "")
    text = text.strip()

    # First try the entire response.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Then search for a JSON object.
    decoder = json.JSONDecoder()

    for i, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


# ============================================================
# SIDE ID VALIDATION
# ============================================================

def _is_valid_side_id_format(side_id):
    """
    Validate the shape of a Side ID.

    Supported examples:
        1683
        6044
        S-3802
        CII-2842

    Explicitly reject station/region-style strings such as:
        S2.LKN.FUEL01
        LKN-FUEL01
    """

    if side_id is None:
        return False

    value = str(side_id).strip().upper()

    if not value:
        return False

    if value in {
        "NULL",
        "NONE",
        "UNKNOWN",
        "N/A",
        "NA",
        "TOO BLURRY TO EXTRACT",
    }:
        return False

    # Station/region codes commonly contain dots.
    if "." in value:
        return False

    # Numeric Side IDs must be exactly 4 digits.
    if value.isdigit():
        return len(value) == 4

    # Examples:
    # S-3802
    # CII-2842
    if re.fullmatch(r"[A-Z]+-\d{4}", value):
        return True

    # More permissive fallback for future alphanumeric IDs.
    if re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", value):
        return True

    return False


# ============================================================
# COORDINATE VALIDATION
# ============================================================

def _parse_coordinate(value):
    """
    Convert a coordinate to float safely.
    """

    if value is None:
        return None

    try:
        text = str(value).strip()
        if not text:
            return None

        # Only allow a normal decimal coordinate.
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return None

        return float(text)

    except Exception:
        return None


def _coordinates_are_valid(latitude, longitude):
    """
    Generic geographic validation.

    We don't hard-code Pakistan bounds here because the extractor
    should remain usable if the project scope changes.
    """

    lat = _parse_coordinate(latitude)
    lon = _parse_coordinate(longitude)

    if lat is None or lon is None:
        return False

    if not (-90 <= lat <= 90):
        return False

    if not (-180 <= lon <= 180):
        return False

    return True


# ============================================================
# NORMALIZE MODEL RESULT
# ============================================================

def _normalize_model_result(data):
    """
    Clean model output without changing the actual meaning.
    """

    if not isinstance(data, dict):
        return None

    side_id = force_string(data.get("side_id"))
    latitude = force_string(data.get("latitude"))
    longitude = force_string(data.get("longitude"))
    mp_number = force_string(data.get("mp_number"))

    # Clean coordinates.
    latitude = clean_coord(latitude)
    longitude = clean_coord(longitude)

    # Clean Side ID.
    side_id = clean_side_id(side_id)

    # Preserve precision up to 8 decimals.
    latitude = truncate_to_8_decimals(latitude)
    longitude = truncate_to_8_decimals(longitude)

    normalized = {
        "side_id": side_id,
        "side_id_clear": bool(data.get("side_id_clear", True)),
        "latitude": latitude,
        "latitude_clear": bool(data.get("latitude_clear", True)),
        "longitude": longitude,
        "longitude_clear": bool(data.get("longitude_clear", True)),
        "mp_number": mp_number,
    }

    return normalized


# ============================================================
# RESULT VALIDATION
# ============================================================

def validate_extraction(data):
    """
    Strong local validation after the vision model.

    Returns:
        (True, "")
    or:
        (False, "reason")
    """

    if not isinstance(data, dict):
        return False, "response is not a JSON object"

    side_id = data.get("side_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not side_id:
        return False, "missing side_id"

    if not latitude:
        return False, "missing latitude"

    if not longitude:
        return False, "missing longitude"

    if not _is_valid_side_id_format(side_id):
        return False, f"invalid side_id format: {side_id!r}"

    # If the model explicitly said the field was unclear,
    # do not trust it.
    if data.get("side_id_clear") is False:
        return False, "side_id marked unclear"

    if data.get("latitude_clear") is False:
        return False, "latitude marked unclear"

    if data.get("longitude_clear") is False:
        return False, "longitude marked unclear"

    if not _coordinates_are_valid(latitude, longitude):
        return False, (
            f"invalid coordinates: "
            f"lat={latitude!r}, lon={longitude!r}"
        )

    return True, ""


# ============================================================
# MODEL CALL
# ============================================================

def _call_model_once(
    image_png_bytes,
    filename,
    rotate_degrees,
    max_tokens=MAX_COMPLETION_TOKENS,
    max_rate_limit_retries=2,
    prompt_override=None,
    filename_hint=None,
):
    """
    One independent vision extraction attempt.
    """

    active_prompt = (
        prompt_override
        if prompt_override is not None
        else PROMPT
    )

    # Filename is deliberately only a hint.
    hint = filename_hint or filename or "unknown.jpg"

    active_prompt = (
        active_prompt
        + "\n\nCURRENT IMAGE FILENAME: "
        + str(hint)
        + "\n"
        + "The filename is ONLY a verification hint. "
          "Never use it instead of reading the watermark."
    )

    if rotate_degrees:
        image_png_bytes = _rotate_isolated_png(
            image_png_bytes,
            rotate_degrees,
        )

    image_png_bytes = _cap_longest_side(
        image_png_bytes,
        max_side=1600,
    )

    for rl_attempt in range(max_rate_limit_retries):

        try:

            image_b64 = base64.b64encode(
                image_png_bytes
            ).decode("utf-8")

            response = client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": active_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/png;base64,"
                                        + image_b64
                                    ),
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],

                # More room than the previous 2000-token limit.
                max_completion_tokens=max_tokens,

                # This task doesn't require heavy reasoning.
                reasoning_effort="low",

                # Force JSON-style output.
                response_format={
                    "type": "json_object"
                },
            )

            choice = response.choices[0]

            text = choice.message.content or ""

            if not text.strip():

                finish_reason = choice.finish_reason

                print(
                    f"   ⚠️ Empty response for {filename} "
                    f"(rotation={rotate_degrees}°, "
                    f"budget={max_tokens}) "
                    f"— finish_reason={finish_reason!r}"
                )

                if finish_reason == "length":

                    # Retry once with more output room.
                    if max_tokens < 5000:
                        print(
                            "   🔁 Retrying because completion "
                            "ended due to length..."
                        )

                        return _call_model_once(
                            image_png_bytes,
                            filename,
                            0,
                            max_tokens=5000,
                            max_rate_limit_retries=1,
                            prompt_override=prompt_override,
                            filename_hint=filename_hint,
                        )

                    return None, "empty_length"

                return None, "empty_other"

            data = _extract_json_object(text)

            if data is None:

                print(
                    f"   ⚠️ Could not parse JSON for "
                    f"{filename} "
                    f"(rotation={rotate_degrees}°)"
                )

                return None, "invalid_json"

            data = _normalize_model_result(data)

            if data is None:
                return None, "invalid_data"

            # Apply your existing quality gate.
            try:
                data = apply_quality_gate(data)
            except Exception as gate_error:
                print(
                    f"   ⚠️ Quality gate warning: "
                    f"{gate_error}"
                )

            if not isinstance(data, dict):
                return None, "quality_gate_rejected"

            # IMPORTANT:
            # Don't remove the *_clear flags here.
            # extractor.py needs them for validation.

            valid, reason = validate_extraction(data)

            if not valid:

                print(
                    f"   ⚠️ Extraction rejected for "
                    f"{filename}: {reason}"
                )

                return None, f"validation_failed:{reason}"

            print(
                f"   [ok] "
                f"Side ID={data.get('side_id')} "
                f"Lat={data.get('latitude')} "
                f"Lon={data.get('longitude')}"
            )

            return data, None

        except Exception as e:

            error_text = str(e)

            print(
                f"   ⚠️ Model error for {filename}: "
                f"{error_text}"
            )

            # Retry transient/rate-limit errors.
            if rl_attempt < max_rate_limit_retries - 1:
                continue

            return None, "api_error"

    return None, "rate_limited"