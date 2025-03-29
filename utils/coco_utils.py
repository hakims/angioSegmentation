"""
COCO Format Utilities for Dr-SAM
--------------------------------
This module provides utilities for working with the COCO (Common Objects in Context) format
for annotations in the Dr-SAM pipeline. It includes functions for:

1. Converting between Dr-SAM's custom bounding box format and COCO format
2. Loading and saving COCO-format annotations
3. Manipulating and querying COCO annotations

The COCO format is a standard for object detection, segmentation, and keypoint annotations.
See: https://cocodataset.org/#format-data
"""

import json
import os
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any


def create_coco_skeleton() -> Dict[str, Any]:
    """
    Create a skeleton COCO dataset structure.
    
    Returns:
        A dictionary with the basic COCO format structure.
    """
    return {
        "info": {
            "year": datetime.datetime.now().year,
            "version": "1.0",
            "description": "Dr-SAM Angiogram Dataset",
            "contributor": "Dr-SAM Pipeline",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "vessel",
                "supercategory": "vessel"
            }
        ],
        "images": [],
        "annotations": []
    }


def drsam_to_coco(
    drsam_boxes: Dict[str, List[List[int]]],
    image_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convert Dr-SAM format bounding boxes to COCO format.
    
    Args:
        drsam_boxes: Dictionary mapping filenames to lists of bounding boxes in Dr-SAM format [x1, y1, x2, y2]
        image_dir: Directory containing the images (needed to get image dimensions)
        output_path: Optional path to save the COCO JSON file
        
    Returns:
        COCO format dictionary
    """
    image_dir = Path(image_dir)
    coco_data = create_coco_skeleton()
    
    image_id = 1
    annotation_id = 1
    
    # Process each image and its bounding boxes
    for filename, boxes in drsam_boxes.items():
        # Skip if there are no bounding boxes
        if not boxes:
            continue
            
        # Get image path and check if it exists
        image_path = image_dir / filename
        if not image_path.exists():
            # Try to find the image with a different extension or in subdirectories
            potential_matches = list(image_dir.glob(f"**/{image_path.stem}.*"))
            if potential_matches:
                image_path = potential_matches[0]
            else:
                print(f"Warning: Could not find image {filename} in {image_dir}")
                continue
        
        # Get image dimensions
        try:
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            height, width = img.shape[:2]
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        
        # Add image to COCO dataset
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": 1,
        })
        
        # Add each bounding box as an annotation
        for box in boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # vessel category
                "bbox": [x1, y1, width, height],  # COCO uses [x, y, width, height]
                "area": width * height,
                "segmentation": [],  # Empty for box-only annotations
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        print(f"COCO annotations saved to {output_path}")
    
    return coco_data


def coco_to_drsam(
    coco_data: Union[Dict[str, Any], str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, List[List[int]]]:
    """
    Convert COCO format annotations to Dr-SAM format.
    
    Args:
        coco_data: COCO format dictionary or path to COCO JSON file
        output_path: Optional path to save the Dr-SAM format JSON file
        
    Returns:
        Dictionary mapping filenames to lists of bounding boxes in Dr-SAM format [x1, y1, x2, y2]
    """
    # Load COCO data if it's a file path
    if isinstance(coco_data, (str, Path)):
        with open(coco_data, 'r') as f:
            coco_data = json.load(f)
    
    # Create mapping from image ID to filename
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    # Create Dr-SAM format dictionary
    drsam_boxes = {}
    
    # Process each annotation
    for annotation in coco_data["annotations"]:
        # Skip if not a bounding box annotation
        if "bbox" not in annotation:
            continue
            
        image_id = annotation["image_id"]
        if image_id not in image_id_to_filename:
            continue
            
        filename = image_id_to_filename[image_id]
        
        # Convert COCO bbox [x, y, width, height] to Dr-SAM [x1, y1, x2, y2]
        x, y, width, height = annotation["bbox"]
        drsam_box = [int(x), int(y), int(x + width), int(y + height)]
        
        # Add to Dr-SAM boxes dictionary
        if filename not in drsam_boxes:
            drsam_boxes[filename] = []
            
        drsam_boxes[filename].append(drsam_box)
    
    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(drsam_boxes, f, indent=2)
            
        print(f"Dr-SAM bounding boxes saved to {output_path}")
    
    return drsam_boxes


def load_coco_annotations(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load COCO annotations from a JSON file.
    
    Args:
        path: Path to COCO JSON file
        
    Returns:
        COCO format dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_coco_annotations(coco_data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save COCO annotations to a JSON file.
    
    Args:
        coco_data: COCO format dictionary
        path: Path to save COCO JSON file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(coco_data, f, indent=2)
        
    print(f"COCO annotations saved to {path}")


def merge_coco_annotations(coco_files: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Merge multiple COCO annotation files into a single COCO dataset.
    
    Args:
        coco_files: List of paths to COCO JSON files
        
    Returns:
        Merged COCO format dictionary
    """
    if not coco_files:
        return create_coco_skeleton()
        
    # Load the first file as the base
    merged_coco = load_coco_annotations(coco_files[0])
    
    # Track the highest IDs used so far
    max_image_id = max([img["id"] for img in merged_coco["images"]]) if merged_coco["images"] else 0
    max_annotation_id = max([ann["id"] for ann in merged_coco["annotations"]]) if merged_coco["annotations"] else 0
    
    # Process each additional file
    for coco_file in coco_files[1:]:
        coco_data = load_coco_annotations(coco_file)
        
        # Create mapping for image IDs
        old_to_new_image_id = {}
        
        # Add images, updating IDs
        for image in coco_data["images"]:
            old_id = image["id"]
            max_image_id += 1
            image["id"] = max_image_id
            old_to_new_image_id[old_id] = max_image_id
            merged_coco["images"].append(image)
        
        # Add annotations, updating IDs
        for annotation in coco_data["annotations"]:
            old_image_id = annotation["image_id"]
            if old_image_id in old_to_new_image_id:
                max_annotation_id += 1
                annotation["id"] = max_annotation_id
                annotation["image_id"] = old_to_new_image_id[old_image_id]
                merged_coco["annotations"].append(annotation)
    
    return merged_coco


def get_annotations_for_image(coco_data: Dict[str, Any], image_id: int) -> List[Dict[str, Any]]:
    """
    Get all annotations for a specific image from COCO data.
    
    Args:
        coco_data: COCO format dictionary
        image_id: Image ID to get annotations for
        
    Returns:
        List of annotation dictionaries
    """
    return [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]


def get_image_id_by_filename(coco_data: Dict[str, Any], filename: str) -> Optional[int]:
    """
    Get the image ID for a specific filename from COCO data.
    
    Args:
        coco_data: COCO format dictionary
        filename: Image filename to get ID for
        
    Returns:
        Image ID or None if not found
    """
    for image in coco_data["images"]:
        if image["file_name"] == filename:
            return image["id"]
    return None


def validate_coco_format(coco_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the structure of COCO format data.
    
    Args:
        coco_data: COCO format dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in coco_data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors
    
    # Check images have required fields
    for i, image in enumerate(coco_data["images"]):
        for field in ["id", "file_name", "width", "height"]:
            if field not in image:
                errors.append(f"Image {i} missing required field: {field}")
    
    # Check annotations have required fields
    for i, ann in enumerate(coco_data["annotations"]):
        for field in ["id", "image_id", "category_id"]:
            if field not in ann:
                errors.append(f"Annotation {i} missing required field: {field}")
        
        # Check bbox format if present
        if "bbox" in ann and (len(ann["bbox"]) != 4 or not all(isinstance(x, (int, float)) for x in ann["bbox"])):
            errors.append(f"Annotation {i} has invalid bbox format")
    
    # Check categories have required fields
    for i, cat in enumerate(coco_data["categories"]):
        for field in ["id", "name"]:
            if field not in cat:
                errors.append(f"Category {i} missing required field: {field}")
    
    return len(errors) == 0, errors 