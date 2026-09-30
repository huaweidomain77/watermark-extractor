"""
core/openai_client.py
OpenAI vision client — one call per attempt, JSON-parsed response.
"""
import os
import re
import json
import time
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
# The key is read from the environment. On your laptop, set it in
# a .env file or export it in the shell. On Railway, set it as an
# environment variable named OPENAI_API_KEY.
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "gpt-5.6-luna"
MAX_COMPLETION_TOKENS = 2000

client = OpenAI(api_key=API_KEY)


# ============================================================
# PROMPTS
# ============================================================
PROMPT = """Preprocessed red/orange watermark isolated as dark text on white.
Fixed line order:
1 datetime  2 region code  3 side_id  4 lat=<v>  5 lng=<v>  6 MP-<v>
(Lines after may show Guarded/fuel — ignore.)

side_id = full identifier after region code, including any letter prefix and
          hyphen (e.g. "1365", "S-3802", "CII-2842").
lat/lng = positive numbers, Pakistan.
IMPORTANT: lat/lng contain many digits. Read every digit carefully —
6 and 0 and 5 look different; do not round, do not drop digits.
If any digit is unclear, set *_clear=false rather than guessing.

Return ONLY this JSON:
{"side_id":<str|null>,"side_id_clear":<bool>,
 "latitude":<str|null>,"latitude_clear":<bool>,
 "longitude":<str|null>,"longitude_clear":<bool>,
 "mp_number":<str|null>}

Set *_clear=false and value=null if ANY character is ambiguous. Never guess.
"""

RAW_CROP_PROMPT = """Find the RED watermark text on this photo. Ignore all
printed form text, handwriting, blue/green stamps, and annotations.

The watermark is 6 lines:
1 datetime (like 2026.08.19.151417)
2 region code (like KH)
3 side_id (like CII-2842, S-3802, 6044)
4 lat=<digits.digits>
5 lng=<digits.digits>
6 MP-<digits>-<digits>

Output the JSON only. No explanation, no markdown.

{"side_id":<str|null>,"side_id_clear":<bool>,
 "latitude":<str|null>,"latitude_clear":<bool>,
 "longitude":<str|null>,"longitude_clear":<bool>,
 "mp_number":<str|null>}

If a character is unclear, set its field null and its *_clear to false.
For the "2G Site ID" / "Side ID" field: read it DIGIT BY DIGIT from left to right.
Do NOT confuse it with MP No., TT No., or FSR No.
If the field contains 4 digits, return all 4 — do not truncate.
Look carefully: '6' has a closed loop at the bottom, '1' is a simple vertical stroke."""


# ============================================================
# ONE API CALL
# ============================================================
def _call_model_once(image_png_bytes, filename, rotate_degrees, max_tokens,
                     max_rate_limit_retries=2, prompt_override=None):
    """
    Makes ONE extraction attempt. prompt_override lets the fallback swap
    in RAW_CROP_PROMPT for a single call.
    """
    active_prompt = prompt_override if prompt_override is not None else PROMPT

    if rotate_degrees:
        image_png_bytes = _rotate_isolated_png(image_png_bytes, rotate_degrees)

    image_png_bytes = _cap_longest_side(image_png_bytes, max_side=1600)

    for rl_attempt in range(max_rate_limit_retries):
        try:
            image_b64 = base64.b64encode(image_png_bytes).decode("utf-8")
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": active_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        }},
                    ],
                }],
                max_completion_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""

            if not text.strip():
                finish_reason = response.choices[0].finish_reason
                print(f"   Empty response for {filename} (rotation={rotate_degrees}°, "
                      f"budget={max_tokens}) — finish_reason: {finish_reason!r}")
                if finish_reason == "length":
                    return None, "empty_length"
                return None, "empty_other"

            text = text.strip().replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                print(f"⚠️ Could not parse JSON for {filename} (rotation={rotate_degrees}°)")
                print(f"   Raw response: {text[:200]!r}")
                return None, "bad_json"

            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse JSON for {filename} (rotation={rotate_degrees}°)")
                print(f"   Raw response: {text[:200]!r}")
                return None, "bad_json"

            data["latitude"] = clean_coord(force_string(data.get("latitude")))
            data["longitude"] = clean_coord(force_string(data.get("longitude")))
            data["side_id"] = clean_side_id(force_string(data.get("side_id")))
            data["mp_number"] = force_string(data.get("mp_number"))

            data["latitude"] = truncate_to_8_decimals(data.get("latitude"))
            data["longitude"] = truncate_to_8_decimals(data.get("longitude"))
            print(f"     [ok at rotation={rotate_degrees}°]")

            data = apply_quality_gate(data)
            data.pop("side_id_clear", None)
            data.pop("latitude_clear", None)
            data.pop("longitude_clear", None)
            return data, None

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait = 30 * (rl_attempt + 1)
                print(f"⏳ Rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            elif "503" in error_str or "unavailable" in error_str.lower():
                wait = 20 * (rl_attempt + 1)
                print(f"⚠️ Server busy, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                raise

    print(f"⚠️ Repeated rate limiting for {filename} (rotation={rotate_degrees}°) — giving up")
    return None, "rate_limited"