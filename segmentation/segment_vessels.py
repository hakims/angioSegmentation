# File: segment_vessels.py
# Version: 0.19 (standardizes terminology to use "boundingBox" consistently)

import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from PIL import Image

from preprocessing.transforms import apply_transform
from dr_sam_core.segmentation.predictors import segmentize, load_model
from dr_sam_core.anomality_detection.skeletonize import skeletonize
from dr_sam_core.anomality_detection.detection_algorithms import find_anomality_points
from dr_sam_core.anomality_detection.utils import remove_pixels_from_skeleton

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "dr_sam_core" / "checkpoints" / "sam_vit_h.pth"

def segment_vessels(
    image_path,
    model_path=DEFAULT_MODEL_PATH,
    apply_anomaly_filter=True,
    model_type="vit_h",
    device=None,
    method="median",
    input_boundingBoxes=None,
):
    """
    Dr-SAM segmentation and anomaly detection pipeline (Section 3.1 + 4.1).
    Fully defers to segmentize() for prompt point generation and refinement.
    Accepts optional list of input_boundingBoxes for multi-ROI support.
    Returns all masks [N, H, W] and preserves skeleton/anomaly analysis.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the SAM model weights
        apply_anomaly_filter: Whether to detect anomalies in vessels
        model_type: SAM model type (vit_h, vit_b, vit_l)
        device: Device to run on (None for auto)
        method: Image transformation method
        input_boundingBoxes: List of bounding boxes for vessel ROIs
        
    Returns:
        Dictionary with masks, skeletons, anomalies, and input points
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply preprocessing transform
    image_np = apply_transform(image_rgb, method=method)

    height, width = image_np.shape[:2]
    if input_boundingBoxes is None:
        input_boundingBoxes = torch.tensor([[0, 0, width, height]])
    elif isinstance(input_boundingBoxes, list):
        input_boundingBoxes = torch.tensor(input_boundingBoxes, dtype=torch.int64)

    if input_boundingBoxes.ndim != 2 or input_boundingBoxes.shape[1] != 4:
        raise ValueError("input_boundingBoxes must have shape [N, 4]")

    # Load SAM predictor
    predictor = load_model(model_type=model_type, model_path=model_path, device=device)

    # Full segmentation using Dr-SAM's 5-point refinement
    result = segmentize(image_np, input_boundingBoxes, predictor)

    masks_tensor = result["masks"]
    if masks_tensor is None or not isinstance(masks_tensor, torch.Tensor) or masks_tensor.numel() == 0:
        print(f"⚠️ No masks returned from segmentize() for: {image_path}")
        return {
            "masks": [],
            "skeleton_raw": [],
            "skeleton": [],
            "anomalies": [],
            "input_points": result["input_points"] if "input_points" in result else []
        }

    masks_np = masks_tensor.cpu().numpy()
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)
    elif masks_np.ndim != 3:
        raise ValueError(f"Unexpected mask shape after conversion: {masks_np.shape}")

    masks_binary = (masks_np > 0.5).astype(np.uint8)

    # Apply skeletonization and anomaly detection if requested
    all_skeletons = []
    all_skeletons_raw = []
    all_anomalies = []

    if apply_anomaly_filter:
        for mask in masks_binary:
            if np.sum(mask) == 0:
                all_skeletons_raw.append(np.zeros_like(mask))
                all_skeletons.append(np.zeros_like(mask))
                all_anomalies.append([])
                continue

            skeleton_raw = skeletonize(mask)
            skeleton_cleaned = remove_pixels_from_skeleton(skeleton_raw, mask)
            anomalies = find_anomality_points(skeleton_cleaned)

            all_skeletons_raw.append(skeleton_raw)
            all_skeletons.append(skeleton_cleaned)
            all_anomalies.append(anomalies)

    return {
        "masks": masks_binary,
        "skeleton_raw": all_skeletons_raw,
        "skeleton": all_skeletons,
        "anomalies": all_anomalies,
        "input_points": result.get("input_points", [])
    }
