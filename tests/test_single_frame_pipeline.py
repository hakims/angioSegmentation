#!/usr/bin/env python
# File: tests/test_single_frame_pipeline.py
# Purpose: Test single frame pipeline processing

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

class TestSingleFramePipeline(BaseTest):
    """Test class for single frame pipeline."""
    
    @pytest.mark.single_frame
    def test_metadata_validation(self, single_frame_case):
        """
        Test metadata validation for single frame.
        
        This test:
        1. Loads metadata from the test case
        2. Validates it against the schema
        3. Verifies required fields are present
        4. Checks attribute values are valid
        """
        # Get test case data
        metadata = single_frame_case["metadata"]
        schema = single_frame_case["schema"]
        case_name = single_frame_case["name"]
        
        # Skip test if no metadata
        if not metadata:
            pytest.skip(f"No metadata found for test case: {case_name}")
        
        # Validate against schema
        is_valid, errors = validate_metadata(metadata, schema)
        assert is_valid, f"Schema validation errors: {errors}"
        
        # Verify required fields
        assert "images" in metadata, "No images found in metadata"
        assert "annotations" in metadata, "No annotations found in metadata"
        
        # Check each annotation
        for ann in metadata["annotations"]:
            # Verify required fields
            assert "id" in ann, "Annotation missing id"
            assert "image_id" in ann, "Annotation missing image_id"
            assert "category_id" in ann, "Annotation missing category_id"
            assert "bbox" in ann, "Annotation missing bbox"
            assert "attributes" in ann, "Annotation missing attributes"
    
    @pytest.mark.single_frame
    def test_annotation_creation(self, single_frame_case, visualization_dir):
        """
        Test annotation visualization and metadata compatibility.
        
        This test:
        1. Loads metadata and image
        2. Directly validates schema compatibility
        3. Visualizes annotations from the COCO format metadata
        """
        # Get test case data
        metadata = single_frame_case["metadata"]
        image = single_frame_case["image"]
        path = single_frame_case["path"]
        case_name = single_frame_case["name"]
        schema = single_frame_case["schema"]
        
        # Skip test if no metadata or image
        if not metadata:
            pytest.skip(f"No metadata found for test case: {case_name}")
        if image is None:
            pytest.skip(f"No image found for test case: {case_name}")
        
        # Log test details
        logger.info(f"Testing annotation creation for case: {case_name}")
        logger.info(f"Image shape: {image.shape if image is not None else 'None'}")
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
        vis_dir = visualization_dir / "single_frame" / case_name
        vis_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created visualization directory: {vis_dir}")
        
        # Visualize annotations directly from COCO metadata
        vis_file = vis_dir / f"{case_name}_annotations.png"
        logger.info(f"Creating visualization at: {vis_file}")
        
        # Use the updated visualize_annotations function
        visualization_success = visualize_annotations(image, metadata, vis_file)
        assert visualization_success, f"Failed to create visualization for {case_name}"
        assert vis_file.exists(), f"Visualization file not created: {vis_file}"
        
        # Log success
        logger.info(f"Successfully created visualization at: {vis_file}")
    
    # ===================================================================
    # The following tests are commented out for later reimplementation
    # These tests involve running the full pipeline and calculating 
    # stenosis metrics, which will be addressed in a future implementation
    # ===================================================================
    
    # @pytest.mark.single_frame
    # def test_pipeline_execution(self, single_frame_case, output_dir):
    #     """
    #     Test pipeline execution on single frame.
    #     
    #     This test:
    #     1. Loads test case data
    #     2. Runs the segmentation pipeline
    #     3. Verifies output files
    #     4. Validates results
    #     """
    #     # Get test case data
    #     metadata = single_frame_case["metadata"]
    #     image = single_frame_case["image"]
    #     path = single_frame_case["path"]
    #     case_name = single_frame_case["name"]
    #     
    #     # Skip test if no metadata or image
    #     if not metadata:
    #         pytest.skip(f"No metadata found for test case: {case_name}")
    #     if image is None:
    #         pytest.skip(f"No image found for test case: {case_name}")
    #         
    #     # Create output directory for this test
    #     test_output_dir = output_dir / "single_frame" / case_name
    #     test_output_dir.mkdir(exist_ok=True, parents=True)
    #     
    #     # Write test image to output directory
    #     image_path = test_output_dir / "test_image.png"
    #     cv2.imwrite(str(image_path), image)
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
    #         # Verify results
    #         mask_file = next(test_output_dir.glob("masks/*.png"), None)
    #         assert mask_file is not None, "No mask files created"
    #         
    #         # Check mask validity
    #         mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    #         assert mask is not None, "Failed to read mask file"
    #         assert mask.shape[0] > 0 and mask.shape[1] > 0, "Mask has invalid dimensions"
    #         assert np.max(mask) > 0, "Mask is completely black (no segmentation)"
    #         
    #     except (ImportError, ModuleNotFoundError) as e:
    #         pytest.skip(f"Could not import buildDrSAM: {e}")
    
    # @pytest.mark.single_frame
    # def test_stenosis_calculation(self, single_frame_case, output_dir):
    #     """
    #     Test stenosis calculation.
    #     
    #     This test:
    #     1. Loads test case data
    #     2. Runs the segmentation pipeline
    #     3. Calculates stenosis
    #     4. Verifies results
    #     """
    #     # Get test case data
    #     metadata = single_frame_case["metadata"]
    #     image = single_frame_case["image"]
    #     path = single_frame_case["path"]
    #     case_name = single_frame_case["name"]
    #     
    #     # Skip test if no metadata or image
    #     if not metadata:
    #         pytest.skip(f"No metadata found for test case: {case_name}")
    #     if image is None:
    #         pytest.skip(f"No image found for test case: {case_name}")
    #     
    #     # Skip if stenosis data not available
    #     has_stenosis = False
    #     for ann in metadata.get("annotations", []):
    #         if ann.get("attributes", {}).get("stenosis_percent") is not None:
    #             has_stenosis = True
    #             break
    #     
    #     if not has_stenosis:
    #         pytest.skip(f"No stenosis data found for test case: {case_name}")
    #     
    #     # Create output directory for this test
    #     test_output_dir = output_dir / "single_frame" / case_name
    #     test_output_dir.mkdir(exist_ok=True, parents=True)
    #     
    #     # Write test image to output directory
    #     image_path = test_output_dir / "test_image.png"
    #     cv2.imwrite(str(image_path), image)
    #     
    #     # Convert COCO to Dr-SAM format
    #     boundingBoxes_dict, attributes_dict = coco_to_drsam(metadata)
    #     
    #     # Run pipeline
    #     try:
    #         from bin.buildDrSAM import main
    #         from segmentation.anomality_detection import calculate_stenosis
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
    #         # Find mask and skeleton files
    #         mask_file = next(test_output_dir.glob("masks/*.png"), None)
    #         skeleton_file = next(test_output_dir.glob("skeletons/*.png"), None)
    #         
    #         assert mask_file is not None, "No mask files created"
    #         assert skeleton_file is not None, "No skeleton files created"
    #         
    #         # Load mask and skeleton
    #         mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    #         skeleton = cv2.imread(str(skeleton_file), cv2.IMREAD_GRAYSCALE)
    #         
    #         assert mask is not None, "Failed to read mask file"
    #         assert skeleton is not None, "Failed to read skeleton file"
    #         
    #         # Calculate stenosis
    #         stenosis_data = calculate_stenosis(mask, skeleton)
    #         
    #         assert stenosis_data is not None, "Failed to calculate stenosis"
    #         assert "stenosis_percent" in stenosis_data, "Stenosis percent not calculated"
    #         assert "stenosis_locations" in stenosis_data, "Stenosis locations not detected"
    #         
    #         # Compare with expected stenosis
    #         expected_stenosis = None
    #         for ann in metadata.get("annotations", []):
    #             if ann.get("attributes", {}).get("stenosis_percent") is not None:
    #                 expected_stenosis = ann["attributes"]["stenosis_percent"]
    #                 break
    #         
    #         assert expected_stenosis is not None, "Expected stenosis not found in metadata"
    #         
    #         # Allow for some tolerance in stenosis calculation
    #         tolerance = 5.0  # 5% tolerance
    #         stenosis_diff = abs(float(stenosis_data["stenosis_percent"]) - float(expected_stenosis))
    #         assert stenosis_diff <= tolerance, f"Stenosis calculation error exceeds tolerance: {stenosis_diff}% > {tolerance}%"
    #         
    #     except (ImportError, ModuleNotFoundError) as e:
    #         pytest.skip(f"Could not import required modules: {e}")
            
    def calculate_stenosis(self, mask):
        """
        Calculate stenosis from mask.
        
        In a real implementation, this would use vessel analysis algorithms.
        For now, this is a placeholder that returns random values.
        """
        # This is a simplified placeholder - in a real implementation,
        # this would analyze the mask to find stenoses.
        if mask is None:
            return []
            
        # Simplified analysis - find contours and measure width variations
        import random
        return [random.uniform(0, 0.8) for _ in range(5)] 