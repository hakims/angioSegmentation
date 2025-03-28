# File: preprocessing/bounding_boxes.py
# Version: 0.05
# Purpose: Enhanced bounding box generation specifically tuned for vessel structures in angiogram images

import cv2
import numpy as np
from .transforms import apply_transform

def generate_boxes_from_vessel_map(vessel_prob_map: np.ndarray, min_size=2000, aspect_ratio_thresh=(0.05, 5.0), max_boxes=3) -> np.ndarray:
    """
    Given a vessel probability map, extract bounding boxes around vessel regions.
    Specifically tuned to isolate vessel structures while avoiding bone and other artifacts.
    Returns boxes as a NumPy array for tensor compatibility.
    """
    vessel_map_gray = cv2.cvtColor(vessel_prob_map, cv2.COLOR_RGB2GRAY) if vessel_prob_map.ndim == 3 else vessel_prob_map

    _, binary = cv2.threshold(vessel_map_gray, np.percentile(vessel_map_gray, 99), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        if area >= min_size and aspect_ratio_thresh[0] <= aspect_ratio <= aspect_ratio_thresh[1]:
            candidate_boxes.append([x, y, x + w, y + h])

    ranked_boxes = sorted(candidate_boxes, key=lambda b: np.mean(vessel_map_gray[b[1]:b[3], b[0]:b[2]]), reverse=True)

    boxes = ranked_boxes[:max_boxes]

    return np.array(boxes)

def generate_boxes_from_image(image: np.ndarray, transform="frangi", min_size=2000, max_boxes=3) -> np.ndarray:
    """
    Given an angiographic image, apply a specifically-tuned vessel-enhancing transform and extract optimized bounding boxes.
    """
    vessel_prob_map = apply_transform(image, transform=transform)
    return generate_boxes_from_vessel_map(vessel_prob_map, min_size=min_size, max_boxes=max_boxes)


def overlay_debug_outputs(image, vessel_map, boxes, save_path_prefix):
    """
    Save debug overlay showing bounding boxes on image and vessel map.
    """
    import os

    vessel_vis = cv2.normalize(vessel_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vessel_vis = cv2.applyColorMap(vessel_vis, cv2.COLORMAP_JET)

    debug_img = image.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    combined = np.hstack((debug_img, vessel_vis))
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    cv2.imwrite(f"{save_path_prefix}_overlay.png", combined)


def batch_generate_boxes(image_dir, output_dir, transform="frangi", min_size=2000, debug=False, max_boxes=3):
    """
    Generate optimized boxes for all images in a folder and optionally save debug overlays.
    """
    import os
    import json
    from glob import glob
    from ..utils.io import read_image, ensure_dir_exists

    image_paths = glob(os.path.join(image_dir, "*.png")) + glob(os.path.join(image_dir, "*.jpg"))
    all_boxes = {}

    for img_path in image_paths:
        image = read_image(img_path)
        vessel_map = apply_transform(image, transform=transform)
        boxes = generate_boxes_from_vessel_map(vessel_map, min_size=min_size, max_boxes=max_boxes)
        fname = os.path.splitext(os.path.basename(img_path))[0]
        all_boxes[fname] = boxes.tolist()

        if debug:
            debug_prefix = os.path.join(output_dir, "debug", fname)
            overlay_debug_outputs(image, vessel_map, boxes, debug_prefix)

    ensure_dir_exists(output_dir)
    with open(os.path.join(output_dir, "boxes.json"), "w") as f:
        json.dump(all_boxes, f, indent=2)

    return all_boxes
