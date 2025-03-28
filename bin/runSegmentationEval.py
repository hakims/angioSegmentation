# File: bin/runSegmentationEval.py
# Purpose: Evaluate and compare Dr-SAM segmentation outputs under different prompt conditions

"""
Compares IoU of raw SAM, naive (full-frame), expert boxes, and auto boxes.
Saves overlays and summary CSV.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import cv2
import numpy as np
import json
import torch
import csv
from segmentation.segment_vessels import segment_vessels
from utils.vis import overlay_mask, compose_comparison_row


def load_ground_truth_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return (mask > 127).astype(np.uint8) if mask is not None else None

def calculate_iou(pred, truth):
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return intersection / union if union > 0 else 0.0

def run_evaluation(image_dir, mask_dir, expert_boxes_path, auto_boxes_path, output_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(expert_boxes_path) as f:
        expert_boxes_dict = json.load(f)
    with open(auto_boxes_path) as f:
        auto_boxes_dict = json.load(f)

    results = []
    for image_path in sorted(image_dir.glob("*.png")):
        name = image_path.name
        gt_mask = load_ground_truth_mask(mask_dir / name)
        if gt_mask is None:
            print(f"⚠️ Missing ground truth for {name}, skipping.")
            continue

        image = cv2.imread(str(image_path))
        H, W = image.shape[:2]
        full_box = [[0, 0, W, H]]

        def to_tensor(box_list):
            return torch.tensor(box_list)

        segmentations = {
            "Raw SAM": segment_vessels(str(image_path), input_boxes=None)["mask"],
            "Naive (1-point)": segment_vessels(str(image_path), input_boxes=to_tensor(full_box))["mask"],
            "Expert": segment_vessels(str(image_path), input_boxes=to_tensor(expert_boxes_dict.get(name, full_box)))["mask"],
            "Auto": segment_vessels(str(image_path), input_boxes=to_tensor(auto_boxes_dict.get(name, full_box)))["mask"],
        }

        ious = {k: calculate_iou(mask, gt_mask) for k, mask in segmentations.items()}
        results.append({"image": name, **ious})

        comparison = compose_comparison_row(
            [overlay_mask(image, segmentations[k]) for k in segmentations],
            list(segmentations.keys())
        )
        cv2.imwrite(str(output_dir / f"{image_path.stem}_compare.png"), comparison)

    # Save IoU results to CSV
    csv_path = output_dir / "iou_scores.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["image"] + list(segmentations.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Saved evaluation results to {csv_path} and overlay images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dr-SAM segmentations using different prompting strategies")
    parser.add_argument("--images", required=True, help="Path to directory of PNG images")
    parser.add_argument("--masks", required=True, help="Path to ground truth binary masks")
    parser.add_argument("--expert-boxes", required=True, help="Path to JSON file with expert bounding boxes")
    parser.add_argument("--auto-boxes", required=True, help="Path to JSON file with auto-generated bounding boxes")
    parser.add_argument("--output", required=True, help="Directory to save output overlays and IoU CSV")
    args = parser.parse_args()

    run_evaluation(args.images, args.masks, args.expert_boxes, args.auto_boxes, args.output)
