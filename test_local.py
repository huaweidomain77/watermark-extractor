"""
test_local.py
Quick sanity check before running the full Gradio app.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from core.extractor import extract_watermark

# Verify API key is loaded
key = os.environ.get("OPENAI_API_KEY", "")
print(f"API key loaded: {'YES (starts with ' + key[:8] + '...)' if key else 'NO — set it in .env'}")

# Pick a test image — put one JPG in test_images/
test_folder = "test_images"
if not os.path.exists(test_folder) or not os.listdir(test_folder):
    print(f"\n⚠️  Put at least one test image in '{test_folder}/' and rerun.")
    raise SystemExit(1)

test_image = os.path.join(test_folder, os.listdir(test_folder)[0])
print(f"\nProcessing: {test_image}\n")

with open(test_image, "rb") as f:
    image_bytes = f.read()

result = extract_watermark(image_bytes, os.path.basename(test_image))

print("\n" + "=" * 50)
print("RESULT")
print("=" * 50)
for k, v in result.items():
    print(f"  {k}: {v!r}")
print("=" * 50)