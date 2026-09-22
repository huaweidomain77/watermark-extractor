"""
core/preprocess.py
Light preprocessing to help MSER find character-shaped regions.
"""
import cv2


def _preprocess_for_mser(arr):
    """
    Light prep so MSER finds character-like regions rather than noise.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.equalizeHist(gray)
    return gray