#!/usr/bin/env python
# File: tests/test_schema_validation.py
# Purpose: Test schema validation for different test cases

import os
import sys
import json
import pytest
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.schema_utils import (
    validate_and_fix_coco_annotations, 
    validate_and_fix_attributes_dict,
    get_default_attributes,
    validate_coco_against_schema,
    get_schema
)
from utils.coco_utils import coco_to_drsam, drsam_to_coco
from tests.test_utils.visualization import visualize_annotations
from tests.test_utils.validation_utils import validate_metadata
from tests.base_test import BaseTest

@pytest.fixture
def visualization_dir():
    """Create and return a directory for visualizations."""
    path = Path("tests/test_outputs/visualizations")
    path.mkdir(exist_ok=True, parents=True)
    return path

def visualize_annotations(image_path, coco_data, output_path):
    """
    Generate visualization of annotations overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        coco_data: COCO format annotation data
        output_path: Path to save the visualization
    """
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Find annotations for this image
    image_id = None
    for img in coco_data.get('images', []):
        if os.path.basename(image_path) == img['file_name']:
            image_id = img['id']
            break
    
    if image_id is None:
        print(f"No matching image ID found for {image_path}")
        return False
    
    # Draw bounding boxes
    colors = {
        "SFA": (1, 0, 0),      # Red
        "CFA": (0, 1, 0),      # Green
        "pop": (0, 0, 1),      # Blue
        "aorta": (1, 1, 0),    # Yellow
        "CIA": (1, 0, 1),      # Magenta
        "EIA": (0, 1, 1),      # Cyan
        "IIA": (0.5, 0.5, 0.5),# Grey
        "AVF": (1, 0.5, 0),    # Orange
        "bypass": (0.5, 0, 1), # Purple
        "innominate": (0, 0.5, 0.5) # Teal
    }
    default_color = (1, 0.3, 0.3)
    
    for annotation in coco_data.get('annotations', []):
        if annotation['image_id'] == image_id:
            # Extract bounding box
            bbox = annotation['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Get attributes
            attributes = annotation.get('attributes', {})
            vessel_id = attributes.get('vessel_id', 'unknown')
            side = attributes.get('side', '')
            segment = attributes.get('segment', '')
            
            # Choose color based on vessel
            color = colors.get(vessel_id, default_color)
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            label = f"{vessel_id} {side} {segment}".strip()
            plt.text(x, y-5, label, color=color, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f"Annotations for {os.path.basename(image_path)}")
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

@pytest.mark.schema
def test_single_frame_validation(single_frame_case, visualization_dir):
    """
    Test the schema validation on the single frame test case.
    
    This test:
    1. Loads a COCO format metadata file
    2. Validates the metadata against the schema
    3. Visualizes the original annotations (if any)
    4. Adds an annotation with invalid attributes if needed
    5. Runs validation to check that errors are detected and fixed
    6. Converts the fixed COCO data to Dr-SAM format
    7. Verifies that attributes now comply with the schema
    8. Visualizes the annotations on the original image
    """
    # Get test case data
    test_dir = single_frame_case["path"]
    metadata = single_frame_case["metadata"]
    image = single_frame_case["image"]
    
    # Check if test directory exists
    assert test_dir.exists(), f"Test directory not found: {test_dir}"
    
    # First, visualize the original metadata to verify what annotations exist
    original_annotations_count = len(metadata.get('annotations', []))
    print(f"\nOriginal metadata has {original_annotations_count} annotations")
    
    # If there are annotations in the original metadata, visualize them
    if original_annotations_count > 0 and image is not None:
        orig_vis_file = visualization_dir / f"{test_dir.name}_original_annotations.png"
        orig_visualization_success = visualize_annotations(image, metadata, orig_vis_file)
        
        if orig_visualization_success:
            print(f"Original annotations visualization saved to: {orig_vis_file}")
    
    # Always save the original COCO data for reference
    orig_coco_file = visualization_dir / f"{test_dir.name}_original_metadata.json"
    with open(orig_coco_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Original metadata saved to: {orig_coco_file}")
    
    # Validate the metadata against the schema
    print("\nValidating metadata against schema...")
    schema_path = Path(__file__).resolve().parents[1] / "data_schema" / "cvat_labeling_schema_2025.03.json"
    
    if schema_path.exists():
        schema_valid, schema_errors = validate_coco_against_schema(metadata, schema_path)
        if schema_errors:
            print(f"Schema validation errors: {schema_errors}")
        assert schema_valid, "Metadata does not conform to schema"
    else:
        print(f"Warning: Schema file not found at {schema_path}, skipping schema validation")
    
    # Make a copy of the original data for testing with invalid annotation
    test_coco_data = metadata.copy()
    
    # If we're adding a test annotation, make it clear in the output
    if original_annotations_count == 0:
        print("\nNo annotations found in original metadata. Adding test annotation with invalid attributes.")
    else:
        print(f"\nAdding test annotation with invalid attributes to verify validation (in addition to {original_annotations_count} existing annotations).")
    
    # Always add an invalid annotation for testing the fix functionality
    # Get the first image ID
    assert len(test_coco_data.get('images', [])) > 0, "No images found in metadata file"
        
    image_id = test_coco_data['images'][0]['id']
    
    # Create an invalid annotation
    invalid_attrs = {
        "vessel_id": "NOT_VALID_VESSEL",  # Invalid vessel_id
        "side": "INVALID_SIDE",           # Invalid side
        "segment": "INVALID_SEGMENT",     # Invalid segment
        "annotator": "test"               # This one is valid
    }
    
    # Add annotation to the COCO data
    test_coco_data['annotations'].append({
        "id": max([a.get("id", 0) for a in test_coco_data.get('annotations', [])], default=0) + 1,
        "image_id": image_id,
        "category_id": 1,  # vessel
        "bbox": [100, 100, 200, 200],
        "area": 40000,
        "segmentation": [],
        "iscrowd": 0,
        "attributes": invalid_attrs
    })
    
    # Test COCO validation
    fixed_coco, errors = validate_and_fix_coco_annotations(test_coco_data)
    
    # Assert that validation found errors
    assert errors, "COCO validation found no issues, which is unexpected for our test data"
    
    # Convert COCO to Dr-SAM format and check for attributes
    boundingBoxes_dict, attributes_dict = coco_to_drsam(fixed_coco)
    
    # Assert that attributes were preserved during conversion
    assert attributes_dict, "No attributes found after conversion"
    
    # Verify that the fixed attributes match schema requirements
    valid_vessel_ids = ["SFA", "aorta", "CFA", "pop", "AVF", "bypass", "CIA", "EIA", "IIA", "innominate"]
    valid_sides = ["L", "R", "B", "N/A"]
    valid_segments = ["", "proximal", "mid", "distal", "AK", "BK"]
    
    for filename, boxes in attributes_dict.items():
        for box_idx, attrs in boxes.items():
            # Check each attribute against the schema
            if "vessel_id" in attrs:
                assert attrs["vessel_id"] in valid_vessel_ids, f"Invalid vessel_id after fixing: {attrs['vessel_id']}"
                
            if "side" in attrs:
                assert attrs["side"] in valid_sides, f"Invalid side after fixing: {attrs['side']}"
                
            if "segment" in attrs:
                assert attrs["segment"] in valid_segments, f"Invalid segment after fixing: {attrs['segment']}"
    
    # Visualize the fixed annotations (original + fixed invalid test annotation)
    if image is not None:
        vis_file = visualization_dir / f"{test_dir.name}_fixed_annotations.png"
        visualization_success = visualize_annotations(image, fixed_coco, vis_file)
        
        if visualization_success:
            print(f"\nFixed annotations visualization saved to: {vis_file}")
    
    # Always save the fixed COCO data for reference
    fixed_coco_file = visualization_dir / f"{test_dir.name}_fixed_metadata.json"
    with open(fixed_coco_file, 'w') as f:
        json.dump(fixed_coco, f, indent=2)
    
    print(f"Fixed metadata saved to: {fixed_coco_file}")

@pytest.mark.schema
def test_create_and_validate_coco(create_test_coco_data):
    """
    Test validation on a programmatically created COCO file with known invalid attributes.
    
    This test:
    1. Creates a COCO file with invalid attributes using the fixture
    2. Validates and fixes the attributes
    3. Confirms the attributes were properly corrected
    """
    # Create test data with invalid attributes
    coco_data = create_test_coco_data(image_id=1, filename="test.png")
    
    # Validate and fix the data
    fixed_coco, errors = validate_and_fix_coco_annotations(coco_data)
    
    # Assert that validation found errors
    assert errors, "COCO validation found no issues, which is unexpected for our test data"
    
    # Convert to Dr-SAM format
    boundingBoxes_dict, attributes_dict = coco_to_drsam(fixed_coco)
    
    # Assert that attributes were preserved during conversion
    assert attributes_dict, "No attributes found after conversion"
    
    # Get the filename and box attributes
    filename = list(attributes_dict.keys())[0]
    box_idx = list(attributes_dict[filename].keys())[0]
    attrs = attributes_dict[filename][box_idx]
    
    # Assert that vessel_id was fixed
    assert attrs["vessel_id"] in ["SFA", "aorta", "CFA", "pop", "AVF", "bypass", "CIA", "EIA", "IIA", "innominate"], \
        f"Invalid vessel_id after fixing: {attrs['vessel_id']}"
    
    # Assert that side was fixed
    assert attrs["side"] in ["L", "R", "B", "N/A"], \
        f"Invalid side after fixing: {attrs['side']}"
    
    # Assert that segment was fixed
    assert attrs["segment"] in ["", "proximal", "mid", "distal", "AK", "BK"], \
        f"Invalid segment after fixing: {attrs['segment']}"

class TestSchemaValidation(BaseTest):
    """Test class for schema validation."""
    
    @pytest.mark.schema
    def test_basic_validation(self, schema_test_case):
        """
        Test basic schema validation.
        
        This test:
        1. Loads metadata from the test case
        2. Validates it against the schema
        3. Verifies required fields are present
        4. Checks attribute values are valid
        """
        # Get test case data
        metadata = schema_test_case["metadata"]
        schema = schema_test_case["schema"]
        
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
    
    @pytest.mark.schema
    def test_format_conversion(self, schema_test_case, output_dir):
        """
        Test format conversion.
        
        This test:
        1. Converts COCO to Dr-SAM format
        2. Converts back to COCO format
        3. Validates the converted data
        """
        # Get test case data
        metadata = schema_test_case["metadata"]
        schema = schema_test_case["schema"]
        case_name = schema_test_case["name"]
        
        # Skip test if no annotations
        if not metadata.get("annotations"):
            pytest.skip("No annotations found in test case")
        
        # Convert COCO to Dr-SAM format
        boundingBoxes_dict, attributes_dict = coco_to_drsam(metadata)
        
        # Verify conversion results
        assert boundingBoxes_dict, "No bounding boxes found after conversion"
        assert attributes_dict, "No attributes found after conversion"
        
        # Create output directory for this test
        test_output_dir = output_dir / "schema_validation" / case_name
        test_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save Dr-SAM format
        drsam_file = test_output_dir / "drsam_format.json"
        with open(drsam_file, 'w') as f:
            json.dump({
                "boundingBoxes": boundingBoxes_dict,
                "attributes": attributes_dict
            }, f, indent=2)
        
        # Convert back to COCO format
        coco_data = drsam_to_coco(
            boundingBoxes_dict,
            schema_test_case["path"],
            attributes=attributes_dict
        )
        
        # Validate converted data
        is_valid, errors = validate_metadata(coco_data, schema)
        assert is_valid, f"Schema validation errors in converted data: {errors}"
        
        # Save converted COCO format
        coco_file = test_output_dir / "coco_format.json"
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
    
    @pytest.mark.schema
    def test_visualization(self, schema_test_case, visualization_dir):
        """
        Test annotation visualization.
        
        This test:
        1. Loads metadata and images
        2. Visualizes annotations
        3. Verifies visualization files are created
        """
        # Get test case data
        metadata = schema_test_case["metadata"]
        images = schema_test_case["images"]
        case_name = schema_test_case["name"]
        
        # Skip test if no images
        if not images:
            pytest.skip("No images found in test case")
        
        # Skip test if no annotations
        if not metadata.get("annotations"):
            pytest.skip("No annotations found in test case")
        
        # Create visualization directory for this test
        vis_dir = visualization_dir / "schema_validation" / case_name
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualize annotations for each image
        for i, image in enumerate(images):
            vis_file = vis_dir / f"image_{i}_annotations.png"
            visualization_success = visualize_annotations(image, metadata, vis_file)
            assert visualization_success, f"Failed to create visualization for image {i}"
            assert vis_file.exists(), f"Visualization file not created: {vis_file}"
    
    @pytest.mark.schema
    def test_edge_cases(self, schema_test_case, validate_test_metadata):
        """
        Test edge cases and error conditions.
        
        This test:
        1. Tests handling of missing required fields
        2. Tests handling of invalid attribute values
        3. Tests handling of malformed metadata
        """
        # Get test case data
        metadata = schema_test_case["metadata"]
        schema = schema_test_case["schema"]
        
        # Skip test if no annotations
        if not metadata.get("annotations"):
            pytest.skip("No annotations found in test case")
        
        # Test missing required fields
        test_metadata = metadata.copy()
        if test_metadata["annotations"]:
            # Remove required field from first annotation
            first_ann = test_metadata["annotations"][0].copy()
            if "bbox" in first_ann:
                del first_ann["bbox"]
                test_metadata["annotations"][0] = first_ann
                
                # Verify validation catches the error
                is_valid, errors = validate_test_metadata(test_metadata, schema)
                assert not is_valid, "Validation should fail for missing required field"
                assert any("bbox" in str(e).lower() for e in errors), \
                    "Error message should mention missing bbox"
        
        # Test invalid attribute values
        test_metadata = metadata.copy()
        if test_metadata["annotations"]:
            # Add invalid attribute value to first annotation
            first_ann = test_metadata["annotations"][0].copy()
            first_ann["attributes"] = first_ann.get("attributes", {}).copy()
            first_ann["attributes"]["invalid_attr"] = "INVALID_VALUE"
            test_metadata["annotations"][0] = first_ann
            
            # Verify validation catches the error
            is_valid, errors = validate_test_metadata(test_metadata, schema)
            assert not is_valid, "Validation should fail for invalid attribute"
            assert any("invalid_attr" in str(e).lower() for e in errors), \
                "Error message should mention invalid attribute" 