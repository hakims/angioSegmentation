# File: preprocessing/bounding_boxes.py
# Version: 0.21 (schema validation and attributes handling)
# Purpose: Enhanced bounding box generation specifically tuned for vessel structures in angiogram images

import json
import numpy as np
import cv2
import os
from pathlib import Path
from .transforms import apply_transform
from typing import Dict, List, Union, Optional, Any
from utils.schema_utils import validate_and_fix_coco_annotations, get_default_attributes, validate_and_fix_attributes_dict

def generate_boundingBoxes_from_vessel_map(vessel_prob_map: np.ndarray, min_size=2000, aspect_ratio_thresh=(0.05, 5.0), max_boundingBoxes=3) -> np.ndarray:
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
    candidate_boundingBoxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        if area >= min_size and aspect_ratio_thresh[0] <= aspect_ratio <= aspect_ratio_thresh[1]:
            candidate_boundingBoxes.append([x, y, x + w, y + h])

    ranked_boundingBoxes = sorted(candidate_boundingBoxes, key=lambda b: np.mean(vessel_map_gray[b[1]:b[3], b[0]:b[2]]), reverse=True)

    boundingBoxes = ranked_boundingBoxes[:max_boundingBoxes]

    return np.array(boundingBoxes)

def generate_boundingBoxes_from_image(image: np.ndarray, method="frangi", min_size=2000, max_boundingBoxes=3) -> np.ndarray:
    """
    Given an angiographic image, apply a specifically-tuned vessel-enhancing transform and extract optimized bounding boxes.
    """
    vessel_prob_map = apply_transform(image, method=method)
    return generate_boundingBoxes_from_vessel_map(vessel_prob_map, min_size=min_size, max_boundingBoxes=max_boundingBoxes)


def overlay_debug_outputs(image, vessel_map, boundingBoxes, save_path_prefix):
    """
    Save debug overlay showing bounding boxes on image and vessel map.
    """
    vessel_vis = cv2.normalize(vessel_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vessel_vis = cv2.applyColorMap(vessel_vis, cv2.COLORMAP_JET)

    debug_img = image.copy()
    if len(debug_img.shape) == 2:
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boundingBoxes:
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    combined = np.hstack((debug_img, vessel_vis))
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    cv2.imwrite(f"{save_path_prefix}_overlay.png", combined)


def load_bounding_boxes(user_provided_path=None, media_folder=None):
    """
    Load bounding boxes from a file or search for them in the media folder.
    
    Args:
        user_provided_path: Path to a JSON file with bounding boxes
        media_folder: Folder to search for bounding boxes JSON files
        
    Returns:
        Tuple containing:
        - Dictionary mapping filenames to lists of bounding boxes
        - Dictionary mapping filenames to attributes (or None if no attributes)
    """
    boundingBoxes_dict = {}
    attributes_dict = {}
    
    # First try to load from user-provided file
    if user_provided_path:
        try:
            user_path = Path(user_provided_path)
            
            # Check if it's in COCO format
            if ".coco" in user_path.name.lower():
                from utils.coco_utils import coco_to_drsam
                
                # Load and validate COCO data
                with open(user_path, 'r') as f:
                    coco_data = json.load(f)
                
                # Validate and fix attributes according to schema
                fixed_coco, errors = validate_and_fix_coco_annotations(coco_data)
                if errors:
                    print(f"Schema validation fixed {len(errors)} issues:")
                    for error in errors[:5]:  # Show first 5 errors
                        print(f"  - {error}")
                    if len(errors) > 5:
                        print(f"  - ... and {len(errors) - 5} more issues")
                
                # Convert validated COCO to DrSAM format
                boundingBoxes_dict, attributes_dict = coco_to_drsam(fixed_coco)
                print(f"Loaded and validated COCO format from {user_path}")
                return boundingBoxes_dict, attributes_dict
            
            # Otherwise assume it's in Dr-SAM format
            with open(user_path, 'r') as f:
                boundingBoxes_dict = json.load(f)
                
            # Check for companion attributes file
            attr_path = user_path.with_suffix('.attributes.json')
            if attr_path.exists():
                with open(attr_path, 'r') as f:
                    attributes_dict = json.load(f)
                
                # Validate and fix attributes according to schema
                if attributes_dict:
                    attributes_dict, validation_stats = validate_and_fix_attributes_dict(attributes_dict)
                    if validation_stats["total_issues"] > 0:
                        print(f"Schema validation fixed issues in {validation_stats['fixed_issues']} attribute sets")
                    
            print(f"Loaded {len(boundingBoxes_dict)} bounding boxes from {user_path}")
            return boundingBoxes_dict, attributes_dict
        except Exception as e:
            print(f"⚠️ Error loading bounding boxes from {user_provided_path}: {e}")
    
    # If no user file or it failed, search in media folder
    if media_folder and not boundingBoxes_dict:
        try:
            media_folder = Path(media_folder)
            
            # Look for boundingBoxes.json in the media folder
            json_files = list(media_folder.glob("**/boundingBoxes.json"))
            if json_files:
                # Use the first one found (prioritize root level if multiple)
                for jf in sorted(json_files, key=lambda x: len(str(x))):
                    with open(jf, 'r') as f:
                        boundingBoxes_dict = json.load(f)
                        
                    # Check for companion attributes file
                    attr_path = jf.with_suffix('.attributes.json')
                    if attr_path.exists():
                        with open(attr_path, 'r') as f:
                            attributes_dict = json.load(f)
                            
                        # Validate and fix attributes according to schema
                        if attributes_dict:
                            attributes_dict, validation_stats = validate_and_fix_attributes_dict(attributes_dict)
                            if validation_stats["total_issues"] > 0:
                                print(f"Schema validation fixed issues in {validation_stats['fixed_issues']} attribute sets")
                            
                    print(f"Found existing bounding boxes in {jf}")
                    return boundingBoxes_dict, attributes_dict
                
            # Look for annotations_coco.json (COCO format) in the media folder
            coco_files = list(media_folder.glob("**/annotations_coco.json"))
            if coco_files:
                # Use the first one found (prioritize root level if multiple)
                for cf in sorted(coco_files, key=lambda x: len(str(x))):
                    with open(cf, 'r') as f:
                        coco_data = json.load(f)
                    
                    # Validate and fix attributes according to schema
                    fixed_coco, errors = validate_and_fix_coco_annotations(coco_data)
                    if errors:
                        print(f"Schema validation fixed {len(errors)} issues:")
                        for error in errors[:5]:  # Show first 5 errors
                            print(f"  - {error}")
                        if len(errors) > 5:
                            print(f"  - ... and {len(errors) - 5} more issues")
                    
                    from utils.coco_utils import coco_to_drsam
                    boundingBoxes_dict, attributes_dict = coco_to_drsam(fixed_coco)
                    print(f"Found and validated COCO annotations in {cf}")
                    return boundingBoxes_dict, attributes_dict
                
                print(f"Searched {len(json_files)} JSON files in {media_folder}, but no valid bounding box data found")
        except Exception as e:
            print(f"Error searching for metadata in media folder: {e}")
    
    return boundingBoxes_dict, attributes_dict


def save_bounding_boxes(boundingBoxes: Dict[str, List[List[int]]], output_path: Union[str, Path], 
                       format: str = "drsam", image_dir: Optional[Union[str, Path]] = None,
                       attributes: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """
    Save bounding boxes to a file in the specified format.
    
    Args:
        boundingBoxes: Dictionary mapping filenames to lists of bounding boxes
        output_path: Path to save the JSON file
        format: Format to save in ('drsam' or 'coco')
        image_dir: Directory containing the images (needed for COCO format)
        attributes: Optional dictionary mapping filenames to annotation attributes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "coco":
        if image_dir is None:
            raise ValueError("image_dir is required for COCO format")
        
        from utils.coco_utils import drsam_to_coco
        drsam_to_coco(boundingBoxes, image_dir, output_path, attributes)
    else:
        # Save in Dr-SAM format
        with open(output_path, 'w') as f:
            json.dump(boundingBoxes, f, indent=2)
        
        # If attributes are provided, save them to a separate file
        if attributes:
            attr_path = output_path.with_suffix('.attributes.json')
            with open(attr_path, 'w') as f:
                json.dump(attributes, f, indent=2)
            print(f"Attributes saved to {attr_path}")
            
        print(f"Dr-SAM bounding boxes saved to {output_path}")


def process_images(image_files, output_folder=None, method="frangi", min_box_size=2000, 
                  max_boundingBoxes=3, existing_boundingBoxes=None, auto_generate=True, 
                  save_debug=True, existing_attributes=None):
    """
    Process a list of images to generate and/or visualize bounding boxes.
    
    Args:
        image_files: List of image files (Path objects or strings)
        output_folder: Output folder for debug visualizations and results
        method: Method to use for vessel enhancement
        min_box_size: Minimum size for keeping auto-generated bounding boxes
        max_boundingBoxes: Maximum number of bounding boxes to generate per image
        existing_boundingBoxes: Dictionary of existing bounding boxes (will be updated)
        auto_generate: Whether to auto-generate bounding boxes
        save_debug: Whether to save debug visualizations
        existing_attributes: Dictionary of existing attributes for annotations
        
    Returns:
        Dictionary mapping filenames to bounding boxes
    """
    # Initialize boundingBoxes dictionary if not provided
    boundingBoxes_dict = existing_boundingBoxes.copy() if existing_boundingBoxes else {}
    
    # Initialize attributes dictionary if provided
    attributes_dict = existing_attributes.copy() if existing_attributes else {}
    
    # Set up debug directory if needed
    debug_boundingBoxes_dir = None
    if output_folder and save_debug:
        debug_boundingBoxes_dir = Path(output_folder) / "debug" / "bounding_boxes"
        debug_boundingBoxes_dir.mkdir(parents=True, exist_ok=True)
    
    # Track newly created attribute sets for validation
    new_files_with_attributes = set()
    
    # Process each image
    for file in image_files:
        file_path = Path(file)
        file_name = file_path.name
        
        # Skip if bounding boxes already exist and auto_generate is False
        if file_name in boundingBoxes_dict and not auto_generate:
            continue
            
        # Read the image
        image = cv2.imread(str(file_path))
        if image is None:
            print(f"⚠️ Skipping unreadable image: {file_name}")
            continue
            
        # Generate bounding boxes if needed
        if auto_generate or file_name not in boundingBoxes_dict:
            boundingBoxes = generate_boundingBoxes_from_image(
                image, 
                method=method, 
                min_size=min_box_size,
                max_boundingBoxes=max_boundingBoxes
            )
            if len(boundingBoxes) > 0:
                boundingBoxes_dict[file_name] = boundingBoxes.tolist()
                
                # Create default attributes for auto-generated boxes if not already present
                if file_name not in attributes_dict:
                    attributes_dict[file_name] = {}
                
                # Add default attributes for each new bounding box
                for box_idx in range(len(boundingBoxes)):
                    if box_idx not in attributes_dict.get(file_name, {}):
                        # Get default attributes from schema
                        default_attrs = get_default_attributes("vessel")
                        
                        # Override with auto-generation specific values
                        default_attrs.update({
                            "vessel_id": "SFA",  # Default to SFA as most common
                            "annotator": "DrSAM",  # Mark as auto-generated
                        })
                        
                        attributes_dict.setdefault(file_name, {})[box_idx] = default_attrs
                        new_files_with_attributes.add(file_name)
        
        # Create debug visualization if requested
        if debug_boundingBoxes_dir and file_name in boundingBoxes_dict:
            debug_prefix = debug_boundingBoxes_dir / file_path.stem
            vessel_map = apply_transform(image, method=method)
            boundingBox_data = boundingBoxes_dict[file_name]
            
            if isinstance(boundingBox_data, list) and len(boundingBox_data) > 0:
                boundingBoxes_np = np.array(boundingBox_data)
                overlay_debug_outputs(image, vessel_map, boundingBoxes_np, str(debug_prefix))
    
    # Validate auto-generated attributes against schema
    if new_files_with_attributes:
        print(f"Validating {len(new_files_with_attributes)} auto-generated attribute sets...")
        
        # Create a subset dictionary with just the new files
        new_attributes_dict = {filename: attributes_dict[filename] for filename in new_files_with_attributes}
        
        # Validate and fix according to schema
        validated_attributes, validation_stats = validate_and_fix_attributes_dict(new_attributes_dict)
        
        # Update the main attributes dictionary with validated values
        for filename, attrs in validated_attributes.items():
            attributes_dict[filename] = attrs
            
        if validation_stats["total_issues"] > 0:
            print(f"Fixed {validation_stats['fixed_issues']} auto-generated attribute sets to comply with schema")
    
    # Save boundingBoxes to JSON files if output folder is provided
    if output_folder:
        output_folder_path = Path(output_folder)
        
        # Save in Dr-SAM format
        drsam_path = output_folder_path / "boundingBoxes.json"
        save_bounding_boxes(boundingBoxes_dict, drsam_path, format="drsam", attributes=attributes_dict)
        
        # Also save in COCO format
        coco_path = output_folder_path / "annotations_coco.json"
        try:
            save_bounding_boxes(
                boundingBoxes_dict, 
                coco_path, 
                format="coco", 
                image_dir=output_folder_path / "frames",
                attributes=attributes_dict
            )
        except Exception as e:
            print(f"Error saving COCO format: {e}")
    
    return boundingBoxes_dict, attributes_dict
