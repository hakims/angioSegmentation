#!/usr/bin/env python
# File: tests/test_utils/visualization.py
# Purpose: Visualization utilities for testing

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_annotations(image, metadata: Dict[str, Any], output_path: Union[str, Path]) -> bool:
    """
    Visualize COCO format annotations on an image.
    
    Args:
        image: Image to visualize annotations on (numpy array or path)
        metadata: COCO format metadata
        output_path: Path to save the visualization
        
    Returns:
        True if successful, False otherwise
    """
    # Load image if path was provided
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            logger.error(f"Failed to load image: {image}")
            return False
    
    # Convert output_path to Path
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Log where we're saving the visualization
    logger.info(f"Saving visualization to: {output_path.absolute()}")
    
    # Make a copy of the image
    vis_image = image.copy()
    
    # If image is grayscale, convert to RGB
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    elif vis_image.shape[2] == 1:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    
    # Draw bounding boxes for each annotation
    for ann in metadata.get("annotations", []):
        # Find annotations for this image
        image_id = ann.get("image_id")
        if not image_id:
            continue
            
        # Get bounding box
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
            
        # Get color based on category
        category_id = ann.get("category_id", 0)
        color = get_category_color(category_id)
        
        # Draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        if "attributes" in ann:
            attrs = ann["attributes"]
            vessel_id = attrs.get("vessel_id", "")
            side = attrs.get("side", "")
            label = f"{vessel_id} {side}"
            
            # Put text above bounding box
            cv2.putText(
                vis_image, 
                label, 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
    
    # Save the visualization
    try:
        cv2.imwrite(str(output_path), vis_image)
        return True
    except Exception as e:
        logger.error(f"Error saving visualization: {e}")
        return False

def get_category_color(category_id: int) -> tuple:
    """
    Get color for a category ID.
    
    Args:
        category_id: Category ID
        
    Returns:
        BGR color tuple
    """
    # Define colors for different categories
    colors = [
        (0, 0, 255),    # Red (BGR)
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]
    
    # Get color based on category ID
    return colors[category_id % len(colors)] 