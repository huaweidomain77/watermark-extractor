"""
test_tesseract.py

Standalone test script to verify Tesseract OCR works correctly
and to see the RAW text it extracts from a real watermark image.

Run this from Railway's Console tab like:

    python test_tesseract.py debug_isolated_watermark.png

This does NOT touch extractor.py or app.py — it's purely
diagnostic, so we can see what Tesseract actually returns
before wiring it into the real pipeline.
"""

import sys
import time

import pytesseract
from PIL import Image


def main():

    if len(sys.argv) < 2:
        print("Usage: python test_tesseract.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Running Tesseract on: {image_path}")

    t0 = time.time()

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    print(f"⏱️ OCR time: {time.time() - t0:.2f}s")

    print("\n" + "=" * 50)
    print("RAW TESSERACT OUTPUT")
    print("=" * 50)

    print(text if text.strip() else "(no text detected)")


if __name__ == "__main__":
    main()