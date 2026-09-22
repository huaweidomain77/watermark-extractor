---
title: Watermark Extractor
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# AI Watermark Extractor

Extract red watermark text (site ID, GPS coordinates, MP number) from
survey photos, then verify each extracted location against a master
site file (300 m threshold).

**How to use:**
1. Upload up to 10 survey images (JPG / PNG).
2. Click **Extract Data** to run the pipeline.
3. (Optional) Upload your master site Excel file and click **Check Distance**
   to validate the extracted coordinates.

**Powered by:** OpenCV (MSER + HSV red mask) + OpenAI GPT-5.6 Luna vision.