"""
test_distance.py
Quick test of the 300m distance verification against master_sites.csv.
"""
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from core.extractor import extract_watermark
from core.distance import load_master_from_excel, validate_against_master, haversine_m


# --- 1. Load the master file ---
print("Loading master_sites.csv ...")
master_df = load_master_from_excel("master_sites.csv")
print(f"✅ Loaded {len(master_df)} master sites")
print(f"   First 3 columns: {list(master_df.columns)}")
print()


# --- 2. Process one test image ---
test_folder = "test_images"
if not os.path.exists(test_folder) or not os.listdir(test_folder):
    print(f"⚠️  No test images in '{test_folder}/'")
    raise SystemExit(1)

test_image = os.path.join(test_folder, os.listdir(test_folder)[0])
print(f"Processing: {test_image}\n")

with open(test_image, "rb") as f:
    image_bytes = f.read()

extracted = extract_watermark(image_bytes, os.path.basename(test_image))
print()
print("Extracted:", extracted)
print()


# --- 3. Run distance verification ---
entries = [{
    "image": os.path.basename(test_image),
    "side_id": extracted.get("side_id"),
    "latitude": extracted.get("latitude"),
    "longitude": extracted.get("longitude"),
    "mp_number": extracted.get("mp_number"),
}]

results = validate_against_master(entries, master_df)


# --- 4. Show the verdict ---
print("=" * 60)
print("DISTANCE VERIFICATION")
print("=" * 60)
for r in results:
    print(f"  Image:          {r['image']}")
    print(f"  Side ID:        {r['side_id']}")
    print(f"  Captured:       {r['latitude']}, {r['longitude']}")
    print(f"  Master:         {r['master_latitude']}, {r['master_longitude']}")
    print(f"  Distance:       {r['distance_m']} m")
    print(f"  Status:         {r['status']}")
    print(f"  Duplicate?:     {r['duplicate_id_in_master']}")
print("=" * 60)