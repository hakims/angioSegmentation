# File: preprocessing/transforms.py
# Purpose: Vessel enhancement filters for angiographic images

import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage.filters import frangi, hessian

def apply_clahe(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_tophat(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return cv2.cvtColor(tophat, cv2.COLOR_GRAY2RGB)

def apply_frangi(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    frangi_out = frangi(gray)
    norm = (255 * frangi_out / frangi_out.max()).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

def apply_hessian(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hessian_out = hessian(gray)
    norm = (255 * hessian_out / hessian_out.max()).astype(np.uint8)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

def apply_median_filter(image_np):
    image_pil = Image.fromarray(image_np)
    image_filtered = image_pil.filter(ImageFilter.ModeFilter(size=7))
    return np.array(image_filtered)

def apply_transform(image_np, transform="median"):
    if transform == "clahe":
        return apply_clahe(image_np)
    elif transform == "tophat":
        return apply_tophat(image_np)
    elif transform == "frangi":
        return apply_frangi(image_np)
    elif transform == "hessian":
        return apply_hessian(image_np)
    else:
        return apply_median_filter(image_np)
