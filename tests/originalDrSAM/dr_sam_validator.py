# File: tests/originalDrSAM/dr_sam_validator.py
# Purpose: Validator for original Dr-SAM dataset

import os
import json
import numpy as np
import cv2
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from .drsam_testing_utils import SegmentationValidator, run_drsam_pipeline


class DrSAMValidator(SegmentationValidator):
    """
    Validator for the original Dr-SAM dataset.
    Handles loading test images, preparing expert bounding boxes,
    and evaluating results against reference masks.
    """
    
    def __init__(self, dataset_path, output_dir="tests/outputs", 
                 results_dir="tests/results", visualize=False,
                 visualize_dir="tests/visualizations", method="median"):
        """
        Initialize the Dr-SAM validator.
        
        Args:
            dataset_path: Path to the Dr-SAM dataset
            output_dir: Directory for Dr-SAM outputs
            results_dir: Directory to save metrics results
            visualize: Whether to save visualization images
            visualize_dir: Directory to save comparison visualizations
            method: Image transformation method to use
        """
        super().__init__(visualize=visualize, results_dir=results_dir, 
                        visualize_dir=visualize_dir)
        
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.method = method
        self.sample_size = None
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Try to load metadata from dataset
        self.metadata = self._load_metadata()
    
    def set_sample_size(self, sample_size):
        """Set the number of images to sample for validation."""
        self.sample_size = sample_size
    
    def _load_metadata(self):
        """Load metadata from dataset if available."""
        metadata_path = self.dataset_path / "metadata.json"
        
        if metadata_path.exists():
            print(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: No metadata.json found at {metadata_path}")
        
        # If no metadata, create a simple one with image paths
        image_files = glob.glob(str(self.dataset_path / "images" / "**" / "*.png"), recursive=True)
        image_files += glob.glob(str(self.dataset_path / "images" / "**" / "*.jpg"), recursive=True)
        
        # Filter out any files in 'masks' directories
        image_files = [f for f in image_files if "masks" not in f]
        
        print(f"Created metadata with {len(image_files)} image paths")
        return {"images": image_files}
    
    @property
    def image_boxes(self):
        """Get bounding boxes for all images from metadata."""
        if not hasattr(self, '_image_boxes'):
            all_images = self.get_test_images()
            self._image_boxes = self.prepare_expert_boundingBoxes(all_images)
        return self._image_boxes
    
    def get_test_images(self, limit=None, shuffle=True):
        """
        Get a list of test images from the dataset.
        
        Args:
            limit: Maximum number of images to return (None for all)
            shuffle: Whether to shuffle the images
            
        Returns:
            List of image paths
        """
        # Use sample_size from instance if set and limit is None
        if limit is None and self.sample_size is not None:
            limit = self.sample_size
            
        images = []
        
        # If metadata has images, use those
        if "images" in self.metadata:
            metadata_images = self.metadata["images"]
            if isinstance(metadata_images, list):
                # Convert relative paths to absolute
                images = [
                    str(Path(img) if os.path.isabs(img) else self.dataset_path / img)
                    for img in metadata_images
                ]
            elif isinstance(metadata_images, dict):
                # If images is a dict, flatten it
                flat_images = []
                for img_id, img_info in metadata_images.items():
                    img_path = img_info.get("path", None)
                    if img_path:
                        flat_images.append(
                            str(Path(img_path) if os.path.isabs(img_path) else self.dataset_path / img_path)
                        )
                images = flat_images
        
        # If no images in metadata, find PNG and JPG files in images directory
        if not images:
            print(f"Looking for images in {self.dataset_path}/images")
            images = glob.glob(str(self.dataset_path / "images" / "**" / "*.png"), recursive=True)
            images += glob.glob(str(self.dataset_path / "images" / "**" / "*.jpg"), recursive=True)
            # Filter out mask files
            images = [img for img in images if "mask" not in img.lower()]
        
        print(f"Found {len(images)} test images")
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(images)
        
        # Apply limit if specified
        if limit and limit > 0:
            images = images[:limit]
            print(f"Using {len(images)} images (limited by sample size)")
        
        return images
    
    def prepare_expert_boundingBoxes(self, image_paths):
        """
        Prepare expert bounding boxes for a list of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping image filenames to bounding boxes
        """
        boundingBoxes_dict = {}
        
        # The metadata is a list of objects, each with image_id and bboxes
        if isinstance(self.metadata, list):
            print(f"Processing metadata list with {len(self.metadata)} entries")
            
            # Create a mapping of image_id to bboxes
            metadata_boxes_by_id = {}
            for item in self.metadata:
                if "image_id" in item and "bboxes" in item:
                    metadata_boxes_by_id[item["image_id"]] = item["bboxes"]
            
            # Match images to bounding boxes
            for img_path in image_paths:
                img_filename = os.path.basename(img_path)
                # Extract image ID from filename (assuming format like "1.png", "2.jpg", etc.)
                img_id_str = os.path.splitext(img_filename)[0]
                try:
                    # Try to convert to int for matching with metadata
                    img_id = int(img_id_str)
                    if img_id in metadata_boxes_by_id:
                        boundingBoxes_dict[img_filename] = metadata_boxes_by_id[img_id]
                except ValueError:
                    # If filename isn't a number, skip it
                    pass
        
        # Legacy format: metadata has a boundingBoxes dictionary
        elif isinstance(self.metadata, dict) and "boundingBoxes" in self.metadata:
            metadata_boundingBoxes = self.metadata["boundingBoxes"]
            
            for img_path in image_paths:
                img_filename = os.path.basename(img_path)
                img_id = os.path.splitext(img_filename)[0]
                
                # Check if boundingBoxes exist for this image
                if img_id in metadata_boundingBoxes:
                    boundingBoxes_dict[img_filename] = metadata_boundingBoxes[img_id]
        
        # Return empty dict if no boundingBoxes found
        if not boundingBoxes_dict:
            print("Warning: No bounding boxes found in metadata for the test images")
        else:
            print(f"Prepared {len(boundingBoxes_dict)} expert bounding boxes")
            
        return boundingBoxes_dict
    
    def get_reference_masks(self, image_path):
        """
        Get reference mask(s) for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of reference mask file paths or None if not found
        """
        img_dir = Path(os.path.dirname(image_path))
        img_filename = os.path.basename(image_path)
        img_id = os.path.splitext(img_filename)[0]
        
        # Try different possible locations for reference masks
        mask_locations = [
            img_dir / "masks",
            img_dir / "mask",
            img_dir.parent / "masks",
            img_dir.parent / "mask",
            self.dataset_path / "masks",
            img_dir / "references",
            img_dir.parent / "references",
        ]
        
        # Check each location
        for mask_dir in mask_locations:
            if mask_dir.exists():
                try:
                    mask_files = self.find_mask_file(mask_dir, img_id)
                    if mask_files:
                        return mask_files
                except FileNotFoundError:
                    continue
        
        return None
    
    def get_generated_masks(self, image_path):
        """
        Get generated mask(s) for an image from the output directory.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            List of generated mask file paths or None if not found
        """
        img_filename = os.path.basename(image_path)
        img_id = os.path.splitext(img_filename)[0]
        
        # Look for generated masks in the output directory
        mask_pattern = self.output_dir / "masks" / f"{img_id}*_mask.png"
        mask_files = glob.glob(str(mask_pattern))
        
        if not mask_files:
            # Also check the root of the output directory
            mask_pattern = self.output_dir / f"{img_id}*_mask.png"
            mask_files = glob.glob(str(mask_pattern))
        
        return mask_files if mask_files else None
    
    def validate_image(self, image_path, invert_reference=False, invert_generated=False):
        """
        Validate a single image by comparing generated and reference masks.
        
        Args:
            image_path: Path to the image
            invert_reference: Whether to invert reference masks
            invert_generated: Whether to invert generated masks
            
        Returns:
            Dictionary with IoU, Dice scores, and visualization path (if enabled)
        """
        img_filename = os.path.basename(image_path)
        img_id = os.path.splitext(img_filename)[0]
        
        # Load original image for visualization
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Get reference masks
        reference_mask_files = self.get_reference_masks(image_path)
        if not reference_mask_files:
            return {"error": f"No reference masks found for {img_id}"}
        
        # Get generated masks
        generated_mask_files = self.get_generated_masks(image_path)
        if not generated_mask_files:
            return {"error": f"No generated masks found for {img_id}"}
        
        # Combine reference masks
        reference_mask = self.combine_masks(reference_mask_files, invert=invert_reference)
        
        # Combine generated masks
        generated_mask = self.combine_masks(generated_mask_files, invert=invert_generated)
        
        # Calculate metrics
        iou = self.calculate_iou(generated_mask, reference_mask)
        dice = self.calculate_dice(generated_mask, reference_mask)
        
        result = {
            "image_id": img_id,
            "filename": img_filename,
            "metrics": {
                "iou": float(iou),
                "dice": float(dice)
            }
        }
        
        # Save visualization if enabled
        if self.visualize:
            vis_path = self.save_comparison_visualization(
                img_id, original_image, reference_mask, generated_mask, iou, dice
            )
            result["visualization"] = str(vis_path)
        
        return result


    def run_validation(self, num_images=5, method="median", 
                      run_pipeline=True, invert_reference=False, 
                      invert_generated=False, save_results=True):
        """
        Run complete validation on a subset of images.
        
        Args:
            num_images: Number of images to validate
            method: Segmentation method to use
            run_pipeline: Whether to run the Dr-SAM pipeline
            invert_reference: Whether to invert reference masks
            invert_generated: Whether to invert generated masks
            save_results: Whether to save results to JSON
            
        Returns:
            Dictionary with validation results
        """
        # Get test images
        test_images = self.get_test_images(limit=num_images)
        if not test_images:
            return {"error": "No test images found"}
        
        # Prepare expert bounding boxes
        boundingBoxes_dict = self.prepare_expert_boundingBoxes(test_images)
        
        # Run Dr-SAM pipeline if requested
        if run_pipeline:
            success = run_drsam_pipeline(
                dataset_path=self.dataset_path,
                method=method,
                output_dir=self.output_dir,
                boundingBoxes_dict=boundingBoxes_dict
            )
            
            if not success:
                return {"error": "Failed to run Dr-SAM pipeline"}
        
        # Validate each image
        results = {
            "num_images": len(test_images),
            "method": method,
            "per_image_results": []
        }
        
        for img_path in test_images:
            try:
                image_result = self.validate_image(
                    img_path, 
                    invert_reference=invert_reference, 
                    invert_generated=invert_generated
                )
                
                results["per_image_results"].append(image_result)
            except Exception as e:
                print(f"Error validating image {img_path}: {e}")
                results["per_image_results"].append({
                    "image_id": os.path.splitext(os.path.basename(img_path))[0],
                    "filename": os.path.basename(img_path),
                    "error": str(e)
                })
        
        # Calculate average metrics
        valid_results = [r for r in results["per_image_results"] 
                        if "metrics" in r and "error" not in r]
        
        if valid_results:
            avg_iou = np.mean([r["metrics"]["iou"] for r in valid_results])
            avg_dice = np.mean([r["metrics"]["dice"] for r in valid_results])
            
            results["average_metrics"] = {
                "avg_iou": float(avg_iou),
                "avg_dice": float(avg_dice)
            }
        
        # Save results if requested
        if save_results:
            results_path = self.save_metrics_results(
                results, prefix=f"validation_{method}"
            )
            results["results_file"] = str(results_path)
        
        return results 

    def evaluate_results(self):
        """Run validation on existing results without running the pipeline."""
        return self.run_validation(
            num_images=self.sample_size,
            method=self.method,
            run_pipeline=False,
            save_results=True
        ) 