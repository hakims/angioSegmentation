#!/usr/bin/env python3
# File: bin/validateCOCO.py
# Version: 0.01
"""
COCO Format Validation Utility for Dr-SAM
-----------------------------------------
This script validates and performs basic analysis on COCO format annotations.
It can compare different annotation files, visualize bounding boxes, and check for common issues.

Run this script with:
    python bin/validateCOCO.py <coco_json_file> [options]
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.coco_utils import (
    load_coco_annotations,
    validate_coco_format,
    get_image_id_by_filename,
    get_annotations_for_image
)


def print_coco_summary(coco_data):
    """
    Print a summary of the COCO annotations.
    
    Args:
        coco_data: COCO format dictionary
    """
    print("\n=== COCO Dataset Summary ===")
    print(f"Dataset: {coco_data.get('info', {}).get('description', 'Unknown')}")
    print(f"Version: {coco_data.get('info', {}).get('version', 'Unknown')}")
    print(f"Images: {len(coco_data.get('images', []))}")
    print(f"Annotations: {len(coco_data.get('annotations', []))}")
    print(f"Categories: {len(coco_data.get('categories', []))}")
    
    # Calculate annotations per image
    image_ids = set(img["id"] for img in coco_data.get("images", []))
    annotated_images = set(ann["image_id"] for ann in coco_data.get("annotations", []))
    
    print(f"Images with annotations: {len(annotated_images)} / {len(image_ids)}")
    
    if coco_data.get("annotations"):
        annotations_per_image = {}
        for ann in coco_data.get("annotations", []):
            image_id = ann["image_id"]
            annotations_per_image[image_id] = annotations_per_image.get(image_id, 0) + 1
        
        if annotations_per_image:
            avg_annotations = sum(annotations_per_image.values()) / len(annotations_per_image)
            max_annotations = max(annotations_per_image.values())
            min_annotations = min(annotations_per_image.values())
            
            print(f"Annotations per image: Avg: {avg_annotations:.1f}, Min: {min_annotations}, Max: {max_annotations}")
        
        # Count annotations with attributes
        annotations_with_attributes = sum(1 for ann in coco_data.get("annotations", []) if "attributes" in ann)
        if annotations_with_attributes:
            print(f"Annotations with attributes: {annotations_with_attributes} / {len(coco_data.get('annotations', []))} ({annotations_with_attributes / len(coco_data.get('annotations', [])) * 100:.1f}%)")
            
            # Count vessel_id values
            vessel_ids = {}
            anomaly_types = {}
            sides = {}
            
            for ann in coco_data.get("annotations", []):
                if "attributes" in ann:
                    attrs = ann["attributes"]
                    if "vessel_id" in attrs:
                        vessel_id = attrs["vessel_id"]
                        vessel_ids[vessel_id] = vessel_ids.get(vessel_id, 0) + 1
                    
                    if "anomaly_type" in attrs:
                        anomaly_type = attrs["anomaly_type"]
                        anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
                        
                    if "side" in attrs:
                        side = attrs["side"]
                        sides[side] = sides.get(side, 0) + 1
            
            # Display top vessel_ids
            if vessel_ids:
                top_vessel_ids = sorted(vessel_ids.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Top vessel_id values:")
                for vessel_id, count in top_vessel_ids:
                    print(f"  - {vessel_id}: {count}")
            
            # Display anomaly_types
            if anomaly_types:
                print("Anomaly types:")
                for anomaly_type, count in sorted(anomaly_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {anomaly_type}: {count}")
                    
            # Display sides
            if sides:
                print("Sides:")
                for side, count in sorted(sides.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {side}: {count}")
    
    print("=== Categories ===")
    for cat in coco_data.get("categories", []):
        print(f"  - {cat.get('id')}: {cat.get('name')} (supercategory: {cat.get('supercategory', 'None')})")


def visualize_annotations(coco_data, image_dir, output_dir, max_images=5):
    """
    Visualize annotations from COCO data.
    
    Args:
        coco_data: COCO format dictionary
        image_dir: Directory containing the images
        output_dir: Directory to save visualizations
        max_images: Maximum number of images to visualize
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get images with annotations
    image_ids = set(ann["image_id"] for ann in coco_data.get("annotations", []))
    image_id_to_info = {img["id"]: img for img in coco_data.get("images", [])}
    
    # Create color map for categories
    category_colors = {}
    for cat in coco_data.get("categories", []):
        category_colors[cat["id"]] = np.random.randint(0, 255, 3).tolist()
    
    # Visualize a sample of images
    count = 0
    for image_id in list(image_ids)[:max_images]:
        if image_id not in image_id_to_info:
            continue
        
        image_info = image_id_to_info[image_id]
        filename = image_info["file_name"]
        
        # Find image file
        image_path = image_dir / filename
        if not image_path.exists():
            # Try to find the image with a different extension or in subdirectories
            potential_matches = list(image_dir.glob(f"**/{image_path.stem}.*"))
            if potential_matches:
                image_path = potential_matches[0]
            else:
                print(f"Warning: Could not find image {filename} in {image_dir}")
                continue
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Get annotations for this image
        annotations = get_annotations_for_image(coco_data, image_id)
        
        # Create visualization
        vis_img = img.copy()
        
        for ann in annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]
            
            if len(bbox) != 4:
                continue
            
            # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = [int(coord) for coord in bbox]
            
            # Get color for this category
            color = category_colors.get(category_id, [0, 255, 0])
            # OpenCV uses BGR
            color = [color[2], color[1], color[0]]
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # Add category label and attributes if present
            category_name = next((cat["name"] for cat in coco_data.get("categories", []) if cat["id"] == category_id), "Unknown")
            label = category_name
            
            # Add vessel_id and anomaly_type if present in attributes
            if "attributes" in ann:
                attrs = ann["attributes"]
                if "vessel_id" in attrs and attrs["vessel_id"]:
                    label += f" - {attrs['vessel_id']}"
                if "anomaly_type" in attrs and attrs["anomaly_type"] and attrs["anomaly_type"] != "none":
                    label += f" ({attrs['anomaly_type']})"
                if "side" in attrs and attrs["side"] and attrs["side"] != "N/A":
                    label += f" {attrs['side']}"
                    
            cv2.putText(vis_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization
        output_path = output_dir / f"{image_path.stem}_vis.png"
        cv2.imwrite(str(output_path), vis_img)
        
        count += 1
        print(f"Saved visualization for {filename} with {len(annotations)} annotations: {output_path}")
    
    print(f"Saved {count} visualizations to {output_dir}")


def compare_coco_files(coco_file1, coco_file2, output_dir=None):
    """
    Compare two COCO annotation files and report differences.
    
    Args:
        coco_file1: Path to first COCO JSON file
        coco_file2: Path to second COCO JSON file
        output_dir: Optional directory to save comparison results
    """
    print(f"\n=== Comparing COCO Files ===")
    print(f"File 1: {coco_file1}")
    print(f"File 2: {coco_file2}")
    
    # Load COCO data
    coco_data1 = load_coco_annotations(coco_file1)
    coco_data2 = load_coco_annotations(coco_file2)
    
    # Compare image counts
    images1 = {img["file_name"]: img for img in coco_data1.get("images", [])}
    images2 = {img["file_name"]: img for img in coco_data2.get("images", [])}
    
    common_images = set(images1.keys()) & set(images2.keys())
    only_in_1 = set(images1.keys()) - set(images2.keys())
    only_in_2 = set(images2.keys()) - set(images1.keys())
    
    print(f"\nImages:")
    print(f"  Total in file 1: {len(images1)}")
    print(f"  Total in file 2: {len(images2)}")
    print(f"  Common: {len(common_images)}")
    print(f"  Only in file 1: {len(only_in_1)}")
    print(f"  Only in file 2: {len(only_in_2)}")
    
    # Compare annotation counts
    annotations1 = coco_data1.get("annotations", [])
    annotations2 = coco_data2.get("annotations", [])
    
    # Group annotations by image_id
    annotations1_by_image = {}
    for ann in annotations1:
        image_id = ann["image_id"]
        if image_id not in annotations1_by_image:
            annotations1_by_image[image_id] = []
        annotations1_by_image[image_id].append(ann)
    
    annotations2_by_image = {}
    for ann in annotations2:
        image_id = ann["image_id"]
        if image_id not in annotations2_by_image:
            annotations2_by_image[image_id] = []
        annotations2_by_image[image_id].append(ann)
    
    # Compare annotation counts by filename
    annotation_diff = {}
    for filename in common_images:
        image_id1 = get_image_id_by_filename(coco_data1, filename)
        image_id2 = get_image_id_by_filename(coco_data2, filename)
        
        count1 = len(annotations1_by_image.get(image_id1, []))
        count2 = len(annotations2_by_image.get(image_id2, []))
        
        if count1 != count2:
            annotation_diff[filename] = (count1, count2)
    
    print(f"\nAnnotations:")
    print(f"  Total in file 1: {len(annotations1)}")
    print(f"  Total in file 2: {len(annotations2)}")
    print(f"  Common images with different annotation counts: {len(annotation_diff)}")
    
    if annotation_diff:
        print("\nSample of annotation count differences:")
        for i, (filename, (count1, count2)) in enumerate(list(annotation_diff.items())[:5]):
            print(f"  {filename}: File 1: {count1}, File 2: {count2}")
    
    # Save comparison results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save list of images only in file 1
        if only_in_1:
            with open(output_dir / "images_only_in_file1.txt", 'w') as f:
                for filename in sorted(only_in_1):
                    f.write(f"{filename}\n")
        
        # Save list of images only in file 2
        if only_in_2:
            with open(output_dir / "images_only_in_file2.txt", 'w') as f:
                for filename in sorted(only_in_2):
                    f.write(f"{filename}\n")
        
        # Save annotation differences
        if annotation_diff:
            with open(output_dir / "annotation_count_diff.txt", 'w') as f:
                f.write("filename,file1_count,file2_count\n")
                for filename, (count1, count2) in sorted(annotation_diff.items()):
                    f.write(f"{filename},{count1},{count2}\n")
        
        print(f"\nSaved comparison results to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="COCO Format Validation Utility")
    
    # Main arguments
    parser.add_argument("coco_file", help="Path to COCO JSON file to validate/analyze")
    parser.add_argument("--image-dir", help="Directory containing the images")
    parser.add_argument("--output-dir", default="coco_validation", help="Directory to save outputs")
    
    # Validation options
    parser.add_argument("--validate", action="store_true", help="Validate COCO format")
    parser.add_argument("--visualize", action="store_true", help="Visualize annotations")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of images to visualize")
    
    # Comparison options
    parser.add_argument("--compare", help="Path to second COCO JSON file to compare against")
    
    args = parser.parse_args()
    
    # Check if COCO file exists
    coco_file = Path(args.coco_file)
    if not coco_file.exists():
        print(f"Error: COCO file not found: {coco_file}")
        sys.exit(1)
    
    # Load COCO data
    coco_data = load_coco_annotations(coco_file)
    
    # Print summary
    print_coco_summary(coco_data)
    
    # Validate format
    if args.validate:
        is_valid, errors = validate_coco_format(coco_data)
        
        print("\n=== Format Validation ===")
        if is_valid:
            print("✅ COCO format is valid")
        else:
            print("❌ COCO format has errors:")
            for error in errors:
                print(f"  - {error}")
    
    # Visualize annotations
    if args.visualize:
        if not args.image_dir:
            print("Error: --image-dir is required for visualization")
            sys.exit(1)
        
        visualize_annotations(
            coco_data=coco_data,
            image_dir=args.image_dir,
            output_dir=Path(args.output_dir) / "visualizations",
            max_images=args.max_images
        )
    
    # Compare COCO files
    if args.compare:
        compare_file = Path(args.compare)
        if not compare_file.exists():
            print(f"Error: Comparison COCO file not found: {compare_file}")
            sys.exit(1)
        
        compare_coco_files(
            coco_file1=coco_file,
            coco_file2=compare_file,
            output_dir=Path(args.output_dir) / "comparison"
        )


if __name__ == "__main__":
    main() 