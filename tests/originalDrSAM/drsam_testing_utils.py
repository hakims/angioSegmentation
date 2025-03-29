# File: tests/originalDrSAM/drsam_testing_utils.py
# Purpose: Testing utilities specific to the original Dr-SAM validation
# Version: 0.02 (adds COCO format support)

import os
import sys
import json
import subprocess
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class SegmentationValidator:
    """Base validator for segmentation results with common functionality."""
    
    def __init__(self, visualize=False, results_dir="tests/results", 
                 visualize_dir="tests/visualizations"):
        """
        Initialize validator with common parameters.
        
        Args:
            visualize: Whether to save visualization images
            results_dir: Directory to save metrics results
            visualize_dir: Directory to save comparison visualizations
        """
        self.visualize = visualize
        self.results_dir = Path(results_dir)
        self.visualize_dir = Path(visualize_dir)
        
        # Create output directories
        if self.visualize:
            self.visualize_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def calculate_iou(self, pred_mask, gt_mask):
        """Calculate IoU between predicted and ground truth masks."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union if union > 0 else 0

    def calculate_dice(self, pred_mask, gt_mask):
        """Calculate Dice coefficient between predicted and ground truth masks."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0
    
    def save_comparison_visualization(self, image_id, original_image, reference_mask, 
                                      our_mask, iou, dice):
        """Save visualization of original image, reference mask, and our mask."""
        vis_path = self.visualize_dir / f"{image_id}_comparison.png"
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Reference mask
        axes[1].imshow(reference_mask, cmap='gray')
        axes[1].set_title("Reference Mask")
        axes[1].axis('off')
        
        # Our mask
        axes[2].imshow(our_mask, cmap='gray')
        axes[2].set_title(f"Generated Mask (IoU: {iou:.3f}, Dice: {dice:.3f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return vis_path
    
    def convert_mask_to_binary(self, mask):
        """Convert mask to binary format (1 for vessel, 0 for background)."""
        if mask.max() > 1:
            # If mask is 0-255, threshold at 127
            return (mask > 127).astype(np.uint8)
        return mask.astype(np.uint8)
    
    def find_mask_file(self, mask_dir, image_id, pattern=None):
        """Find mask file for a given image ID."""
        if pattern is None:
            pattern = f"{image_id}*.png"
        
        mask_pattern = mask_dir / pattern
        mask_files = list(glob.glob(str(mask_pattern)))
        
        if not mask_files:
            raise FileNotFoundError(f"No mask files found for image {image_id}")
        
        return mask_files
    
    def load_mask(self, mask_path, invert=False):
        """
        Load mask from file path and convert to binary.
        
        Args:
            mask_path: Path to mask file
            invert: Whether to invert the mask (if vessels are white on black)
        
        Returns:
            Binary mask as numpy array (1 for vessel, 0 for background)
        """
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Could not read mask file: {mask_path}")
        
        # Convert to binary based on inversion setting
        if invert:
            # If mask has vessels as white (255) on black background (0)
            binary_mask = (mask > 127).astype(np.uint8)
        else:
            # If mask has vessels as black (0) on white background (255)
            binary_mask = (mask < 127).astype(np.uint8)
        
        return binary_mask
    
    def combine_masks(self, mask_files, invert=False):
        """
        Combine multiple mask files using logical OR.
        
        Args:
            mask_files: List of mask file paths
            invert: Whether to invert masks before combining
            
        Returns:
            Combined binary mask
        """
        combined_mask = None
        
        for mask_file in mask_files:
            binary_mask = self.load_mask(mask_file, invert)
            
            if combined_mask is None:
                combined_mask = binary_mask
            else:
                # Combine masks with logical OR
                combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
        
        return combined_mask
    
    def save_metrics_results(self, results, prefix="validation_results"):
        """
        Save metrics results to JSON file.
        
        Args:
            results: Dictionary containing metrics results
            prefix: Prefix for the output file name
            
        Returns:
            Path to the saved results file
        """
        results_path = self.results_dir / f"{prefix}_{len(results.get('per_image_results', []))}_images.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results_path


def get_drsam_command(dataset_path, method="median", output_dir=None, 
                     boundingBoxes_file=None, verbose=False, output_format="both"):
    """
    Build the command to run the Dr-SAM pipeline.
    
    Args:
        dataset_path: Path to the dataset
        method: Method to use (default: median)
        output_dir: Custom output directory
        boundingBoxes_file: Path to user-provided boundingBoxes JSON file
        verbose: Whether to enable verbose output
        output_format: Format for output annotations ('drsam', 'coco', or 'both')
        
    Returns:
        List of command arguments
    """
    # Get path to buildDrSAM.py
    repo_root = Path(__file__).resolve().parents[2]
    buildDrSAM_path = repo_root / "bin" / "buildDrSAM.py"
    
    # Start with basic command
    cmd = [sys.executable, str(buildDrSAM_path), str(dataset_path)]
    
    # Add method if specified
    if method and method != "median":
        cmd.extend(["--method", method])
    
    # Add output directory if specified
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    
    # Add user-provided boundingBoxes if specified
    if boundingBoxes_file:
        cmd.extend(["--user-provided-boundingBoxes", str(boundingBoxes_file)])
        print(f"Adding bounding boxes file to command: {boundingBoxes_file}")
        # Disable auto boundingBox generation when we have user-provided boundingBoxes
        cmd.append("--no-auto-boundingBox")
    else:
        print("No bounding boxes file provided. Auto-generation will be used if enabled.")
    
    # Add output format if not both
    if output_format != "both":
        cmd.extend(["--output-format", output_format])
    
    # Add quiet flag if not verbose
    if not verbose:
        cmd.append("--quiet")
    
    return cmd


def run_drsam_pipeline(dataset_path, method="median", output_dir="tests/outputs", 
                      boundingBoxes_dict=None, verbose=False, output_format="both"):
    """
    Run the Dr-SAM pipeline with the given parameters.
    
    Args:
        dataset_path: Path to the dataset
        method: Method to use (default: median)
        output_dir: Custom output directory
        boundingBoxes_dict: Dictionary of bounding boxes or path to JSON file
        verbose: Whether to enable verbose output
        output_format: Format for output annotations ('drsam', 'coco', or 'both')
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Handle bounding boxes
    temp_boundingBoxes_file = None
    boundingBoxes_file = None
    
    if boundingBoxes_dict:
        if isinstance(boundingBoxes_dict, dict):
            # Save bounding boxes to a temporary file
            temp_boundingBoxes_file = output_dir / "temp_boundingBoxes.json"
            print(f"Saving bounding boxes to temporary file: {temp_boundingBoxes_file}")
            with open(temp_boundingBoxes_file, 'w') as f:
                json.dump(boundingBoxes_dict, f, indent=2)
            boundingBoxes_file = str(temp_boundingBoxes_file)
        elif isinstance(boundingBoxes_dict, str):
            # If it's already a file path
            boundingBoxes_file = boundingBoxes_dict
            print(f"Using existing bounding boxes file: {boundingBoxes_file}")
    
    try:
        # Build and run command
        cmd = get_drsam_command(
            dataset_path=dataset_path,
            method=method,
            output_dir=output_dir,
            boundingBoxes_file=boundingBoxes_file,
            verbose=verbose,
            output_format=output_format
        )
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    
    finally:
        # Clean up temporary files
        if temp_boundingBoxes_file and temp_boundingBoxes_file.exists():
            try:
                os.remove(temp_boundingBoxes_file)
                print(f"Removed temporary file: {temp_boundingBoxes_file}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {e}")


def convert_original_drsam_to_coco(original_format_path, image_dir, output_path=None):
    """
    Convert original Dr-SAM format (array of objects) to COCO format.
    
    The original Dr-SAM format is an array of objects with:
    - image_id: The image ID
    - bboxes: List of bounding boxes in [x1, y1, x2, y2] format
    
    Args:
        original_format_path: Path to original Dr-SAM format JSON file
        image_dir: Directory containing the images (needed for dimensions)
        output_path: Optional path to save COCO format JSON file
        
    Returns:
        COCO format dictionary
    """
    # Import COCO utilities
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    
    from utils.coco_utils import create_coco_skeleton
    
    # Load original Dr-SAM format
    with open(original_format_path, 'r') as f:
        original_data = json.load(f)
    
    # Create COCO skeleton
    coco_data = create_coco_skeleton()
    
    # Track IDs
    image_id = 1
    annotation_id = 1
    
    # Convert each entry
    for item in original_data:
        original_image_id = item.get("image_id")
        bboxes = item.get("bboxes", [])
        
        if not original_image_id or not bboxes:
            continue
        
        # Find the image file
        image_path = None
        potential_matches = list(Path(image_dir).glob(f"**/*{original_image_id}*"))
        
        if potential_matches:
            # Find the one that's most likely an image
            for match in potential_matches:
                if match.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                    image_path = match
                    break
            
            if not image_path:
                image_path = potential_matches[0]
        
        if not image_path:
            print(f"Warning: Could not find image for ID {original_image_id}")
            continue
        
        # Get image dimensions
        try:
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
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "original_id": original_image_id,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": 1,
        })
        
        # Add each bounding box as an annotation
        for box in bboxes:
            if len(box) != 4:
                continue
                
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # vessel category
                "bbox": [x1, y1, box_width, box_height],  # COCO uses [x, y, width, height]
                "area": box_width * box_height,
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


def convert_original_drsam_to_drsam_format(original_format_path, output_path=None):
    """
    Convert original Dr-SAM format (array of objects) to our Dr-SAM format (filename to boxes map).
    
    Args:
        original_format_path: Path to original Dr-SAM format JSON file
        output_path: Optional path to save our Dr-SAM format JSON file
        
    Returns:
        Dictionary mapping filenames to lists of bounding boxes
    """
    # Load original Dr-SAM format
    with open(original_format_path, 'r') as f:
        original_data = json.load(f)
    
    # Convert to our format
    drsam_boxes = {}
    
    for item in original_data:
        image_id = item.get("image_id")
        bboxes = item.get("bboxes", [])
        
        if not image_id or not bboxes:
            continue
        
        # Use image_id as filename (our pipeline will handle finding the actual file)
        filename = f"{image_id}.png"
        drsam_boxes[filename] = bboxes
    
    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(drsam_boxes, f, indent=2)
            
        print(f"Dr-SAM format boxes saved to {output_path}")
    
    return drsam_boxes 