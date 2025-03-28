# File: utils/vis.py
# Purpose: Visualization helpers for debugging and presentation

import cv2
import numpy as np

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    image_copy = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
    return image_copy

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def draw_point(image, point, color=(0, 0, 255), radius=4):
    x, y = point
    return cv2.circle(image.copy(), (x, y), radius, color, -1)

def compose_comparison_row(images, labels, font_scale=0.5):
    out_images = []
    for img, label in zip(images, labels):
        img_copy = img.copy()
        cv2.putText(img_copy, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        out_images.append(img_copy)
    return np.hstack(out_images)

def highlight_anomalies(image, anomaly_points, anomaly_types=None, radius=5):
    '''
    Overlays colored circles at anomaly locations.
    
    Params:
        image (np.ndarray): Original image (H x W x 3)
        anomaly_points (List[Tuple[int, int]]): Coordinates of anomalies [(x1, y1), ...]
        anomaly_types (List[str], optional): List of types ("stenosis", "aneurysm") matching each point
        radius (int): Circle radius
    Returns:
        np.ndarray: Annotated image
    '''
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    color_map = {
        "stenosis": (0, 0, 255),    # Red
        "aneurysm": (255, 0, 0),    # Blue
        "default": (0, 255, 255)    # Yellow fallback
    }

    for i, (x, y) in enumerate(anomaly_points):
        anomaly_type = anomaly_types[i] if anomaly_types and i < len(anomaly_types) else "default"
        color = color_map.get(anomaly_type, color_map["default"])
        cv2.circle(image, (int(x), int(y)), radius, color, thickness=-1)

    return image