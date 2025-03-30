# File: tests/test_utils/drsam_testing_utils.py
# Purpose: Utilities for Dr-SAM testing and validation

import os
import sys
import json
import subprocess
import numpy as np
import cv2
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from segmentation.segmentationPipeline import segment_and_save_outputs
from datetime import datetime


class SegmentationValidator:
    """Base class for segmentation validation."""
    
    def __init__(self, visualize=False, results_dir="tests/results", 
                 visualize_dir="tests/visualizations"):
        self.visualize = visualize
        self.results_dir = Path(results_dir)
        self.visualize_dir = Path(visualize_dir)
        
        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True, parents=True)
        if self.visualize:
            self.visualize_dir.mkdir(exist_ok=True, parents=True)
    
    def calculate_iou(self, pred_mask, gt_mask):
        """Calculate IoU between predicted and ground truth masks."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union if union > 0 else 0

    def calculate_dice(self, pred_mask, gt_mask):
        """Calculate Dice coefficient between predicted and ground truth masks."""
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        return (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0
    
    def prepare_expert_boundingBoxes(self, image_paths, metadata):
        """Prepare expert bounding boxes for a list of images."""
        boundingBoxes_dict = {}
        
        # The metadata is a list of objects, each with image_id and bboxes
        if isinstance(metadata, list):
            print(f"Processing metadata list with {len(metadata)} entries")
            
            # Create a mapping of image_id to bboxes
            metadata_boxes_by_id = {}
            for item in metadata:
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
        elif isinstance(metadata, dict) and "boundingBoxes" in metadata:
            metadata_boundingBoxes = metadata["boundingBoxes"]
            
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
        """Get reference mask(s) for an image."""
        # Look for masks in the same directory as the image
        img_dir = os.path.dirname(image_path)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Try different mask naming patterns
        mask_patterns = [
            f"{img_name}_mask.png",
            f"{img_name}_mask.jpg",
            f"{img_name}.mask.png",
            f"{img_name}.mask.jpg"
        ]
        
        masks = []
        for pattern in mask_patterns:
            mask_path = os.path.join(img_dir, pattern)
            if os.path.exists(mask_path):
                masks.append(mask_path)
        
        if not masks:
            print(f"Warning: No reference masks found for {image_path}")
        
        return masks
    
    def validate_images(self, image_paths, metadata):
        """Validate a list of images."""
        results = {
            "total_images": len(image_paths),
            "processed_images": 0,
            "failed_images": 0,
            "iou_scores": [],
            "errors": []
        }
        
        for img_path in image_paths:
            try:
                # Get reference masks
                ref_masks = self.get_reference_masks(img_path)
                if not ref_masks:
                    results["failed_images"] += 1
                    results["errors"].append(f"No reference masks found for {img_path}")
                    continue
                
                # Run segmentation
                output_dir = Path(self.results_dir) / os.path.splitext(os.path.basename(img_path))[0]
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # Get bounding boxes for this image
                img_filename = os.path.basename(img_path)
                boundingBoxes = self.prepare_expert_boundingBoxes([img_path], metadata)
                img_boxes = boundingBoxes.get(img_filename, [])
                
                # Run segmentation pipeline
                segment_and_save_outputs(
                    frames=[img_path],
                    output_root=output_dir,
                    method="median",
                    frame_to_boundingBoxes={img_filename: img_boxes}
                )
                
                # Calculate IoU
                generated_mask = output_dir / "masks" / f"{os.path.splitext(img_filename)[0]}_mask.png"
                if not generated_mask.exists():
                    results["failed_images"] += 1
                    results["errors"].append(f"Failed to generate mask for {img_path}")
                    continue
                
                # Load masks and calculate IoU
                ref_mask = cv2.imread(str(ref_masks[0]), cv2.IMREAD_GRAYSCALE)
                gen_mask = cv2.imread(str(generated_mask), cv2.IMREAD_GRAYSCALE)
                
                if ref_mask is None or gen_mask is None:
                    results["failed_images"] += 1
                    results["errors"].append(f"Failed to load masks for {img_path}")
                    continue
                
                # Ensure masks are binary
                ref_mask = (ref_mask > 0).astype(np.uint8)
                gen_mask = (gen_mask > 0).astype(np.uint8)
                
                # Calculate IoU
                intersection = np.sum(np.logical_and(ref_mask, gen_mask))
                union = np.sum(np.logical_or(ref_mask, gen_mask))
                iou = intersection / union if union > 0 else 0
                
                results["iou_scores"].append(iou)
                results["processed_images"] += 1
                
                # Save visualization if requested
                if self.visualize:
                    self._save_visualization(img_path, ref_mask, gen_mask, iou)
                
            except Exception as e:
                results["failed_images"] += 1
                results["errors"].append(f"Error processing {img_path}: {str(e)}")
        
        # Calculate average metrics
        if results["processed_images"] > 0:
            results["average_iou"] = np.mean(results["iou_scores"])
            results["std_iou"] = np.std(results["iou_scores"])
        
        return results
    
    def _save_visualization(self, image_path, ref_mask, gen_mask, iou):
        """Save visualization of comparison."""
        # Load original image
        img = cv2.imread(str(image_path))
        if img is None:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Reference mask
        axes[1].imshow(ref_mask, cmap="gray")
        axes[1].set_title("Reference Mask")
        axes[1].axis("off")
        
        # Generated mask
        axes[2].imshow(gen_mask, cmap="gray")
        axes[2].set_title(f"Generated Mask (IoU: {iou:.3f})")
        axes[2].axis("off")
        
        # Save figure
        vis_path = self.visualize_dir / f"{os.path.splitext(os.path.basename(image_path))[0]}_comparison.png"
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close()


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
            self._image_boxes = self.prepare_expert_boundingBoxes(all_images, self.metadata)
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
    
    def get_generated_masks(self, image_path):
        """
        Get generated mask(s) for an image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of mask file paths
        """
        img_filename = os.path.basename(image_path)
        img_id = os.path.splitext(img_filename)[0]
        
        # Look for masks in the output directory
        masks_dir = self.output_dir / img_id / "masks"
        
        if not masks_dir.exists():
            return []
        
        # Find all mask files
        mask_files = glob.glob(str(masks_dir / "*.png"))
        return mask_files
    
    def validate_image(self, image_path, invert_reference=False, invert_generated=False):
        """
        Validate a single image by comparing generated masks with reference masks.
        
        Args:
            image_path: Path to the image
            invert_reference: Whether to invert reference masks
            invert_generated: Whether to invert generated masks
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "image_path": image_path,
            "success": False,
            "metrics": {}
        }
        
        try:
            # Get reference masks
            reference_masks = self.get_reference_masks(image_path)
            if not reference_masks:
                results["error"] = "No reference masks found"
                return results
            
            # Get generated masks
            generated_masks = self.get_generated_masks(image_path)
            if not generated_masks:
                results["error"] = "No generated masks found"
                return results
            
            # Load reference mask (using first one if multiple)
            ref_mask_path = reference_masks[0]
            ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Load generated mask (using first one if multiple)
            gen_mask_path = generated_masks[0]
            gen_mask = cv2.imread(gen_mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Validate mask loading
            if ref_mask is None or gen_mask is None:
                results["error"] = "Failed to load masks"
                return results
            
            # Convert to binary
            ref_mask = (ref_mask > 127).astype(np.uint8)
            gen_mask = (gen_mask > 127).astype(np.uint8)
            
            # Invert if requested
            if invert_reference:
                ref_mask = 1 - ref_mask
            if invert_generated:
                gen_mask = 1 - gen_mask
            
            # Calculate metrics
            iou = self.calculate_iou(gen_mask, ref_mask)
            dice = self.calculate_dice(gen_mask, ref_mask)
            
            results["success"] = True
            results["metrics"] = {
                "iou": float(iou),
                "dice": float(dice)
            }
            
            # Save visualization if requested
            if self.visualize:
                img = cv2.imread(image_path)
                vis_path = self.visualize_dir / f"{os.path.splitext(os.path.basename(image_path))[0]}_comparison.png"
                self.save_comparison_visualization(img, ref_mask, gen_mask, iou, dice, str(vis_path))
                results["visualization"] = str(vis_path)
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def save_comparison_visualization(self, img, ref_mask, gen_mask, iou, dice, output_path):
        """Save visualization of comparison."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Reference mask
        axes[1].imshow(ref_mask, cmap="gray")
        axes[1].set_title("Reference Mask")
        axes[1].axis("off")
        
        # Generated mask
        axes[2].imshow(gen_mask, cmap="gray")
        axes[2].set_title(f"Generated Mask (IoU: {iou:.3f}, Dice: {dice:.3f})")
        axes[2].axis("off")
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    def run_validation(self, num_images=5, method="median", 
                      run_pipeline=True, invert_reference=False, 
                      invert_generated=False, save_results=True):
        """
        Run the complete validation process.
        
        Args:
            num_images: Number of images to validate
            method: Image transformation method to use
            run_pipeline: Whether to run the pipeline
            invert_reference: Whether to invert reference masks
            invert_generated: Whether to invert generated masks
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with validation results
        """
        # Set method to use
        self.method = method
        
        # Get test images
        images = self.get_test_images(limit=num_images)
        
        if not images:
            print("No test images found")
            return None
        
        print(f"Running validation on {len(images)} images")
        
        # Prepare expert bounding boxes
        boundingBoxes_dict = self.image_boxes
        
        # Run pipeline if requested
        if run_pipeline:
            print(f"Running Dr-SAM pipeline with method: {method}")
            
            # Run pipeline for each image
            for img_path in images:
                img_filename = os.path.basename(img_path)
                img_boxes = boundingBoxes_dict.get(img_filename, [])
                
                # Create output directory
                output_dir = self.output_dir / os.path.splitext(img_filename)[0]
                output_dir.mkdir(exist_ok=True, parents=True)
                
                try:
                    run_drsam_pipeline(
                        image_path=img_path,
                        output_dir=output_dir,
                        method=method
                    )
                except Exception as e:
                    print(f"Error running pipeline for {img_path}: {e}")
        
        # Validate images
        results = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "num_images": len(images),
            "metrics": {
                "iou": [],
                "dice": []
            },
            "per_image_results": []
        }
        
        for img_path in images:
            print(f"Validating {img_path}")
            img_result = self.validate_image(
                image_path=img_path,
                invert_reference=invert_reference,
                invert_generated=invert_generated
            )
            
            results["per_image_results"].append(img_result)
            
            if img_result["success"]:
                results["metrics"]["iou"].append(img_result["metrics"]["iou"])
                results["metrics"]["dice"].append(img_result["metrics"]["dice"])
        
        # Calculate average metrics
        if results["metrics"]["iou"]:
            results["metrics"]["avg_iou"] = float(np.mean(results["metrics"]["iou"]))
            results["metrics"]["std_iou"] = float(np.std(results["metrics"]["iou"]))
        
        if results["metrics"]["dice"]:
            results["metrics"]["avg_dice"] = float(np.mean(results["metrics"]["dice"]))
            results["metrics"]["std_dice"] = float(np.std(results["metrics"]["dice"]))
        
        # Save results if requested
        if save_results:
            results_file = self.results_dir / f"validation_results_{method}_{len(images)}_images.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_file}")
        
        return results
    
    def evaluate_results(self):
        """
        Evaluate existing results without running the pipeline.
        
        Returns:
            Dictionary with evaluation results
        """
        return self.run_validation(
            run_pipeline=False,
            num_images=self.sample_size
        )


def run_drsam_pipeline(dataset_path, method="median", output_dir="tests/outputs", 
                      boundingBoxes_dict=None, verbose=False, output_format="both"):
    """
    Run the Dr-SAM pipeline on a dataset.
    
    Args:
        dataset_path: Path to the dataset
        method: Image transformation method to use
        output_dir: Directory for outputs
        boundingBoxes_dict: Dictionary mapping filenames to bounding boxes
        verbose: Whether to print verbose output
        output_format: Format for output annotations
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get path to buildDrSAM.py
        repo_root = Path(__file__).resolve().parents[2]
        buildDrSAM_path = repo_root / "bin" / "buildDrSAM.py"
        
        if not buildDrSAM_path.exists():
            print(f"Error: buildDrSAM.py not found at {buildDrSAM_path}")
            return False
        
        # If it's a Pathlib object, convert to string
        if isinstance(dataset_path, Path):
            dataset_path = str(dataset_path)
            
        if isinstance(output_dir, Path):
            output_dir = str(output_dir)
        
        # Import directly for in-process execution
        sys.path.append(str(repo_root))
        from bin.buildDrSAM import main
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Run pipeline
        main(
            input_path=dataset_path,
            use_fps=False,
            method=method,
            quiet=not verbose,
            auto_boundingBox=boundingBoxes_dict is None,
            boundingBox_method="frangi" if boundingBoxes_dict is None else None,
            min_boundingBox_size=2000,
            user_provided_boundingBoxes=boundingBoxes_dict,
            user_provided_attributes=None,
            boundingBox_file=None,
            output_dir=output_dir,
            force_extract=False,
            output_format=output_format,
            validate_schema=True
        )
        
        return True
        
    except Exception as e:
        print(f"Error running Dr-SAM pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False 