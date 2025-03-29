# File: preprocessing/bounding_boxes.py
# Version: 0.09 (adds COCO format support)
# Purpose: Enhanced bounding box generation specifically tuned for vessel structures in angiogram images

import cv2
import numpy as np
import json
import os
from pathlib import Path
from .transforms import apply_transform
from typing import Dict, List, Union, Optional

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
    Load bounding boxes from user-provided file or search for metadata in media folder.
    Supports both Dr-SAM native format and COCO format.
    
    Args:
        user_provided_path: Optional path to user-provided JSON file
        media_folder: Folder containing media files, to search for metadata
        
    Returns:
        Dictionary of bounding boxes, or empty dict if none found
    """
    boundingBoxes_dict = {}
    
    # 1. Try user-provided path first (highest priority)
    if user_provided_path:
        try:
            user_path = Path(user_provided_path)
            if user_path.exists():
                with open(user_path, 'r') as f:
                    data = json.load(f)
                
                # Check if this is COCO format or Dr-SAM format
                if "images" in data and "annotations" in data:
                    # This is COCO format, convert to Dr-SAM format
                    from utils.coco_utils import coco_to_drsam
                    boundingBoxes_dict = coco_to_drsam(data)
                    print(f"Loaded COCO format annotations from {user_path} and converted to Dr-SAM format")
                else:
                    # Assume Dr-SAM format
                    boundingBoxes_dict = data
                    print(f"Loaded {len(boundingBoxes_dict)} bounding boxes from user-provided file: {user_path}")
                
                return boundingBoxes_dict
            else:
                print(f"Warning: User-provided bounding box file not found: {user_path}")
        except Exception as e:
            print(f"Error loading user-provided bounding box file: {e}")
    
    # 2. Look for any JSON files in the media folder
    if media_folder:
        try:
            media_path = Path(media_folder)
            json_files = list(media_path.glob("*.json"))
            
            if json_files:
                # Try each JSON file to see if it contains bounding box data
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Check if COCO format
                        if isinstance(data, dict) and "images" in data and "annotations" in data:
                            from utils.coco_utils import coco_to_drsam
                            boundingBoxes_dict = coco_to_drsam(data)
                            print(f"Found COCO format annotations in {json_file.name}")
                            return boundingBoxes_dict
                            
                        # Check if Dr-SAM format (dictionary mapping filenames to bounding boxes)
                        elif isinstance(data, dict) and any(
                            isinstance(v, list) and len(v) > 0 and 
                            all(isinstance(coord, (int, float)) for item in v for coord in item)
                            for v in data.values()
                        ):
                            boundingBoxes_dict = data
                            print(f"Found Dr-SAM format bounding box metadata in {json_file.name} ({len(boundingBoxes_dict)} entries)")
                            return boundingBoxes_dict
                    except Exception as e:
                        print(f"Error parsing {json_file.name}: {e}")
                        continue
                
                print(f"Searched {len(json_files)} JSON files in {media_folder}, but no valid bounding box data found")
        except Exception as e:
            print(f"Error searching for metadata in media folder: {e}")
    
    return boundingBoxes_dict


def save_bounding_boxes(boundingBoxes: Dict[str, List[List[int]]], output_path: Union[str, Path], 
                       format: str = "drsam", image_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Save bounding boxes to a file in the specified format.
    
    Args:
        boundingBoxes: Dictionary mapping filenames to lists of bounding boxes
        output_path: Path to save the JSON file
        format: Format to save in ('drsam' or 'coco')
        image_dir: Directory containing the images (needed for COCO format)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "coco":
        if image_dir is None:
            raise ValueError("image_dir is required for COCO format")
        
        from utils.coco_utils import drsam_to_coco
        drsam_to_coco(boundingBoxes, image_dir, output_path)
    else:
        # Save in Dr-SAM format
        with open(output_path, 'w') as f:
            json.dump(boundingBoxes, f, indent=2)
        
        print(f"Dr-SAM bounding boxes saved to {output_path}")


def process_images(image_files, output_folder=None, method="frangi", min_box_size=2000, 
                  max_boundingBoxes=3, existing_boundingBoxes=None, auto_generate=True, save_debug=True):
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
        
    Returns:
        Dictionary mapping filenames to bounding boxes
    """
    # Initialize boundingBoxes dictionary if not provided
    boundingBoxes_dict = existing_boundingBoxes.copy() if existing_boundingBoxes else {}
    
    # Set up debug directory if needed
    debug_boundingBoxes_dir = None
    if output_folder and save_debug:
        debug_boundingBoxes_dir = Path(output_folder) / "debug" / "bounding_boxes"
        debug_boundingBoxes_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        # Create debug visualization if requested
        if debug_boundingBoxes_dir and file_name in boundingBoxes_dict:
            debug_prefix = debug_boundingBoxes_dir / file_path.stem
            vessel_map = apply_transform(image, method=method)
            boundingBox_data = boundingBoxes_dict[file_name]
            
            if isinstance(boundingBox_data, list) and len(boundingBox_data) > 0:
                boundingBoxes_np = np.array(boundingBox_data)
                overlay_debug_outputs(image, vessel_map, boundingBoxes_np, str(debug_prefix))
    
    # Save boundingBoxes to JSON files if output folder is provided
    if output_folder:
        output_folder_path = Path(output_folder)
        
        # Save in Dr-SAM format
        drsam_path = output_folder_path / "boundingBoxes.json"
        save_bounding_boxes(boundingBoxes_dict, drsam_path, format="drsam")
        
        # Also save in COCO format
        coco_path = output_folder_path / "annotations_coco.json"
        try:
            save_bounding_boxes(
                boundingBoxes_dict, 
                coco_path, 
                format="coco", 
                image_dir=output_folder_path / "frames"
            )
        except Exception as e:
            print(f"Error saving COCO format: {e}")
    
    return boundingBoxes_dict
