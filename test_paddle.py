"""
test_paddle.py

Standalone test script to verify PaddleOCR works correctly
and to see the RAW text it extracts from a real watermark image.

Run this from Railway's Console tab (or locally if pip works
for you) like:

    python test_paddle.py test_images/your_image.jpg

This does NOT touch extractor.py or app.py — it's purely
diagnostic, so we can see what PaddleOCR actually returns
before wiring it into the real pipeline.
"""

import sys
import time

from paddleocr import PaddleOCR


def main():

    if len(sys.argv) < 2:
        print("Usage: python test_paddle.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Loading PaddleOCR (this downloads models on first run)...")

    t0 = time.time()

    # use_angle_cls handles rotated text; lang='en' for English/numeric
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False,
    )

    print(f"⏱️ PaddleOCR init (incl. model download if first run): {time.time() - t0:.2f}s")

    print(f"\nRunning OCR on: {image_path}")

    t1 = time.time()

    result = ocr.ocr(image_path, cls=True)

    print(f"⏱️ OCR inference time: {time.time() - t1:.2f}s")

    print("\n" + "=" * 50)
    print("RAW OCR OUTPUT")
    print("=" * 50)

    if not result or result[0] is None:
        print("No text detected.")
        return

    for line in result[0]:
        box, (text, confidence) = line
        print(f"Text: {text!r}   Confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()