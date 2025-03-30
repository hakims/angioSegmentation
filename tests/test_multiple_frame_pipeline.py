#!/usr/bin/env python
# File: tests/test_multiple_frame_pipeline.py
# Purpose: Test multiple frame pipeline processing

import os
import sys
import json
import pytest
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.schema_utils import validate_coco_against_schema
from utils.coco_utils import coco_to_drsam, drsam_to_coco
from tests.test_utils.visualization import visualize_annotations
from tests.test_utils.validation_utils import validate_metadata
from tests.base_test import BaseTest

class TestMultipleFramePipeline(BaseTest):
    """Test class for multiple frame pipeline."""
    
    @pytest.mark.multiple_frame
    def test_metadata_validation(self, multiple_frame_case):
        """
        Test metadata validation for multiple frames.
        
        This test:
        1. Loads metadata from the test case
        2. Validates it against the schema
        3. Verifies required fields are present
        4. Checks attribute values are valid
        """
        # Get test case data
        metadata = multiple_frame_case["metadata"]
        schema = multiple_frame_case["schema"]
        case_name = multiple_frame_case["name"]
        
        # Skip test if no metadata
        if not metadata:
            pytest.skip(f"No metadata found for test case: {case_name}")
        
        # Validate against schema
        is_valid, errors = validate_metadata(metadata, schema)
        assert is_valid, f"Schema validation errors: {errors}"
        
        # Verify required fields
        assert "images" in metadata, "No images found in metadata"
        assert "annotations" in metadata, "No annotations found in metadata"
        
        # Check each image
        for img in metadata["images"]:
            # Verify required fields
            assert "id" in img, "Image missing id"
            assert "file_name" in img, "Image missing file_name"
            assert "width" in img, "Image missing width"
            assert "height" in img, "Image missing height"
        
        # Check each annotation
        for ann in metadata["annotations"]:
            # Verify required fields
            assert "id" in ann, "Annotation missing id"
            assert "image_id" in ann, "Annotation missing image_id"
            assert "category_id" in ann, "Annotation missing category_id"
            assert "bbox" in ann, "Annotation missing bbox"
            assert "attributes" in ann, "Annotation missing attributes"
    
    @pytest.mark.multiple_frame
    def test_annotation_creation(self, multiple_frame_case, visualization_dir):
        """
        Test annotation visualization and metadata compatibility for multiple frames.
        
        This test:
        1. Loads metadata and images
        2. Directly validates schema compatibility
        3. Visualizes annotations for each frame from the COCO format metadata
        """
        # Get test case data
        metadata = multiple_frame_case["metadata"]
        images = multiple_frame_case["images"]
        image_paths = multiple_frame_case.get("image_paths", [])
        path = multiple_frame_case["path"]
        case_name = multiple_frame_case["name"]
        schema = multiple_frame_case["schema"]
        
        # Skip test if no metadata or images
        if not metadata:
            pytest.skip(f"No metadata found for test case: {case_name}")
        if not images:
            pytest.skip(f"No images found for test case: {case_name}")
        
        # Log test details
        logger.info(f"Testing annotation creation for case: {case_name}")
        logger.info(f"Number of images: {len(images)}")
        logger.info(f"Visualization directory: {visualization_dir}")
        
        # Directly validate schema compatibility
        is_valid, errors = validate_metadata(metadata, schema)
        assert is_valid, f"Schema validation errors: {errors}"
        
        # Verify COCO format requirements
        assert "images" in metadata, "Missing 'images' array in COCO metadata"
        assert "annotations" in metadata, "Missing 'annotations' array in COCO metadata"
        assert "categories" in metadata, "Missing 'categories' array in COCO metadata"
        
        # Check image IDs match up with annotation image_ids
        image_ids = {img["id"] for img in metadata["images"]}
        for ann in metadata["annotations"]:
            assert ann["image_id"] in image_ids, f"Annotation references unknown image_id: {ann['image_id']}"
        
        # Create visualization directory for this test
        vis_dir = visualization_dir / "multiple_frame" / case_name
        vis_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created visualization directory: {vis_dir}")
        
        # Get mapping of file names to images
        image_dict = {}
        for i, img_data in enumerate(metadata["images"]):
            if i < len(images):
                image_dict[img_data["file_name"]] = images[i]
        
        # For each image, create a visualization
        count = 0
        for file_name, image in image_dict.items():
            # Skip if image is None
            if image is None:
                logger.warning(f"Skipping visualization for {file_name}: image is None")
                continue
                
            # Create temporary COCO data with only this image
            img_id = next((img["id"] for img in metadata["images"] if img["file_name"] == file_name), None)
            if img_id is None:
                logger.warning(f"Skipping visualization for {file_name}: no matching image ID")
                continue
                
            single_image_coco = {
                "images": [img for img in metadata["images"] if img["file_name"] == file_name],
                "annotations": [ann for ann in metadata["annotations"] if ann["image_id"] == img_id],
                "categories": metadata["categories"]
            }
            
            # Visualize annotations
            vis_file = vis_dir / f"{case_name}_{count}_annotations.png"
            logger.info(f"Creating visualization for frame {count} at: {vis_file}")
            
            # Use the updated visualize_annotations function
            visualization_success = visualize_annotations(image, single_image_coco, vis_file)
            assert visualization_success, f"Failed to create visualization for {file_name}"
            assert vis_file.exists(), f"Visualization file not created: {vis_file}"
            
            # Log success
            logger.info(f"Successfully created visualization for frame {count} at: {vis_file}")
            count += 1
        
        # Verify that at least one visualization was created
        assert count > 0, f"No visualizations created for case: {case_name}"
    
    # ===================================================================
    # The following tests are commented out for later reimplementation
    # These tests involve running the full pipeline and tracking vessels
    # across multiple frames, which will be addressed in a future implementation
    # ===================================================================
    
    # @pytest.mark.multiple_frame
    # def test_pipeline_execution(self, multiple_frame_case, output_dir):
    #     """
    #     Test pipeline execution on multiple frames.
    #     
    #     This test:
    #     1. Loads test case data
    #     2. Runs the segmentation pipeline
    #     3. Verifies output files
    #     4. Validates results
    #     """
    #     # Get test case data
    #     metadata = multiple_frame_case["metadata"]
    #     images = multiple_frame_case["images"]
    #     path = multiple_frame_case["path"]
    #     case_name = multiple_frame_case["name"]
    #     
    #     # Skip test if no metadata or images
    #     if not metadata:
    #         pytest.skip(f"No metadata found for test case: {case_name}")
    #     if not images:
    #         pytest.skip(f"No images found for test case: {case_name}")
    #         
    #     # Create output directory for this test
    #     test_output_dir = output_dir / "multiple_frame" / case_name
    #     test_output_dir.mkdir(exist_ok=True, parents=True)
    #     
    #     # Write test images to output directory
    #     image_paths = []
    #     for i, img_data in enumerate(metadata["images"]):
    #         if i < len(images):
    #             image_path = test_output_dir / img_data["file_name"]
    #             cv2.imwrite(str(image_path), images[i])
    #             image_paths.append(image_path)
    #     
    #     # Convert COCO to Dr-SAM format
    #     boundingBoxes_dict, attributes_dict = coco_to_drsam(metadata)
    #     
    #     # Run pipeline
    #     try:
    #         from bin.buildDrSAM import main
    #         
    #         # Run pipeline with test data
    #         main(
    #             input_path=str(test_output_dir),
    #             output_dir=str(test_output_dir),
    #             user_provided_boundingBoxes=boundingBoxes_dict,
    #             user_provided_attributes=attributes_dict,
    #             validate_schema=True
    #         )
    #         
    #         # Verify output files
    #         assert (test_output_dir / "masks").exists(), "Masks directory not created"
    #         
    #         # Verify results for each image
    #         for i, img_path in enumerate(image_paths):
    #             img_filename = img_path.name
    #             img_stem = img_path.stem
    #             
    #             # Check for mask
    #             mask_file = test_output_dir / "masks" / f"{img_stem}_mask.png"
    #             assert mask_file.exists(), f"Mask not created for {img_filename}"
    #             
    #             # Check mask validity
    #             mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    #             assert mask is not None, f"Failed to read mask for {img_filename}"
    #             assert mask.shape[0] > 0 and mask.shape[1] > 0, f"Mask has invalid dimensions for {img_filename}"
    #             assert np.max(mask) > 0, f"Mask is completely black for {img_filename}"
    #     
    #     except (ImportError, ModuleNotFoundError) as e:
    #         pytest.skip(f"Could not import buildDrSAM: {e}")
    # 
    # @pytest.mark.multiple_frame
    # def test_vessel_tracking(self, multiple_frame_case, output_dir):
    #     """
    #     Test vessel tracking across multiple frames.
    #     
    #     This test:
    #     1. Loads test case data
    #     2. Runs the segmentation pipeline
    #     3. Verifies vessel tracking
    #     4. Validates consistency across frames
    #     """
    #     # Get test case data
    #     metadata = multiple_frame_case["metadata"]
    #     images = multiple_frame_case["images"]
    #     path = multiple_frame_case["path"]
    #     case_name = multiple_frame_case["name"]
    #     
    #     # Skip test if no metadata or images
    #     if not metadata:
    #         pytest.skip(f"No metadata found for test case: {case_name}")
    #     if not images or len(images) < 2:
    #         pytest.skip(f"At least 2 images required for vessel tracking test")
    #         
    #     # Create output directory for this test
    #     test_output_dir = output_dir / "multiple_frame" / f"{case_name}_tracking"
    #     test_output_dir.mkdir(exist_ok=True, parents=True)
    #     
    #     # Write test images to output directory
    #     image_paths = []
    #     for i, img_data in enumerate(metadata["images"]):
    #         if i < len(images):
    #             image_path = test_output_dir / img_data["file_name"]
    #             cv2.imwrite(str(image_path), images[i])
    #             image_paths.append(image_path)
    #     
    #     # Convert COCO to Dr-SAM format
    #     boundingBoxes_dict, attributes_dict = coco_to_drsam(metadata)
    #     
    #     try:
    #         from bin.buildDrSAM import main
    #         from utils.tracking_utils import track_vessels
    #         
    #         # Run pipeline with test data
    #         main(
    #             input_path=str(test_output_dir),
    #             output_dir=str(test_output_dir),
    #             user_provided_boundingBoxes=boundingBoxes_dict,
    #             user_provided_attributes=attributes_dict,
    #             validate_schema=True
    #         )
    #         
    #         # Get mask paths
    #         mask_paths = sorted(list((test_output_dir / "masks").glob("*.png")))
    #         assert len(mask_paths) >= 2, "Need at least 2 masks for tracking"
    #         
    #         # Track vessels across frames
    #         tracking_results = track_vessels(mask_paths)
    #         
    #         # Verify tracking results
    #         assert tracking_results is not None, "Failed to track vessels"
    #         assert "tracked_vessels" in tracking_results, "No tracked vessels in results"
    #         assert len(tracking_results["tracked_vessels"]) > 0, "No vessels were tracked"
    #         
    #         # Check for vessel continuity
    #         for vessel_id, frames in tracking_results["tracked_vessels"].items():
    #             assert len(frames) > 0, f"Vessel {vessel_id} has no frames"
    #             
    #             # For continuous tracking, verify frame sequence
    #             frame_nums = sorted(list(frames.keys()))
    #             for i in range(1, len(frame_nums)):
    #                 assert frame_nums[i] - frame_nums[i-1] <= 2, f"Gap in tracking for vessel {vessel_id}"
    #         
    #         # Save tracking results
    #         tracking_file = test_output_dir / "tracking_results.json"
    #         with open(tracking_file, 'w') as f:
    #             json.dump(tracking_results, f, indent=2)
    #             
    #     except (ImportError, ModuleNotFoundError) as e:
    #         pytest.skip(f"Could not import required modules: {e}")
    #     except Exception as e:
    #         pytest.skip(f"Error in vessel tracking: {e}")
    
    # Helper methods
    def _get_image_by_filename(self, images_data, images, filename):
        """Get image by filename from the list of images."""
        for i, img_data in enumerate(images_data):
            if img_data["file_name"] == filename and i < len(images):
                return images[i]
        return None 