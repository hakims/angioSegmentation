# File: bin/generateBoundingBoxes.py
# Purpose: CLI wrapper for generating bounding boxes using preprocessing/bounding_boxes.py

"""
Auto-generate vessel bounding boxes from angiogram frames or images.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import cv2
import json
import os

from preprocessing.bounding_boxes import generate_boxes_from_image, draw_boxes_on_image

def main(input_path, output_dir, transform="frangi", min_size=300, visualize=False):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_image = input_path.is_file()
    output_json = output_dir / "boxes.json"
    boxes_dict = {}

    image_paths = [input_path] if is_image else sorted(input_path.glob("*.png"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️ Skipping unreadable image: {image_path.name}")
            continue

        boxes = generate_boxes_from_image(image, transform=transform, min_size=min_size)
        boxes_dict[image_path.name] = boxes

        if visualize:
            overlay = draw_boxes_on_image(image, boxes)
            overlay_path = output_dir / f"{image_path.stem}_boxes.png"
            cv2.imwrite(str(overlay_path), overlay)

    with open(output_json, "w") as f:
        json.dump(boxes_dict, f, indent=2)

    print(f"✅ Saved bounding boxes to: {output_json}")
    if visualize:
        print(f"🖼️ Saved overlay previews to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-generate bounding boxes from angiogram images")
    parser.add_argument("input", type=str, help="Path to image or folder of PNGs")
    parser.add_argument("output", type=str, help="Directory to save output boxes and overlays")
    parser.add_argument("--transform", type=str, default="frangi", help="Image transform to use (frangi, clahe, etc.)")
    parser.add_argument("--min-size", type=int, default=300, help="Minimum box area to keep")
    parser.add_argument("--visualize", action="store_true", help="Save box overlay images for preview")
    args = parser.parse_args()

    main(args.input, args.output, transform=args.transform, min_size=args.min_size, visualize=args.visualize)