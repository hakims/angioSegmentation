# File: tests/test_original_drsam.py
# Purpose: Tests for validating against the original Dr-SAM implementation

import os
import json
import numpy as np
import cv2
from pathlib import Path
import pytest
from tests.test_utils.drsam_testing_utils import DrSAMValidator, run_drsam_pipeline
from tests.test_utils.file_utils import find_image_files, find_all_image_files

class TestOriginalDrSAM:
    """Test class for original Dr-SAM validation."""
    
    @pytest.mark.original_drsam
    def test_load_dataset(self, test_root):
        """Test loading the original Dr-SAM dataset."""
        dataset_path = test_root / "original_drsam"
        if not dataset_path.exists():
            pytest.skip("Original Dr-SAM dataset not found")
        
        validator = DrSAMValidator(dataset_path)
        assert validator.dataset_path.exists(), "Dataset path does not exist"
        
        # Check for image folder
        image_dir = validator.dataset_path / "images"
        assert image_dir.exists(), "Images directory not found"
        
        # Find images using our utility
        images = find_all_image_files(image_dir)
        assert len(images) > 0, "No images found in dataset"
        
        # Check metadata
        metadata = validator.load_metadata()
        assert metadata, "Failed to load metadata"
        assert "images" in metadata, "No images in metadata"
        assert "annotations" in metadata, "No annotations in metadata"
        
        # Check that we can load expert annotations
        expert_annotations = validator.load_expert_annotations()
        assert expert_annotations, "Failed to load expert annotations"
        
        print(f"Found {len(images)} images and {len(metadata['annotations'])} annotations")
    
    @pytest.mark.original_drsam
    def test_compare_with_drsam(self, test_root):
        """
        Test comparing our segmentation with original Dr-SAM.
        
        This test:
        1. Loads the original Dr-SAM dataset
        2. Runs our segmentation pipeline on the same images
        3. Compares our results with the expert annotations
        """
        dataset_path = test_root / "original_drsam"
        if not dataset_path.exists():
            pytest.skip("Original Dr-SAM dataset not found")
        
        # Initialize validator
        validator = DrSAMValidator(dataset_path)
        
        # Run on a sample of images (for faster testing)
        image_files = find_all_image_files(validator.dataset_path / "images")
        sample_size = min(5, len(image_files))
        sample_images = image_files[:sample_size]
        
        # Run our pipeline
        results = run_drsam_pipeline(
            images=sample_images,
            output_dir=test_root / "outputs" / "drsam_comparison",
            method="frangi"
        )
        
        assert results, "Pipeline execution failed"
        assert "masks" in results, "No masks generated"
        assert "metrics" in results, "No metrics calculated"
        
        # Validate metrics
        metrics = results["metrics"]
        if metrics:
            assert "dice" in metrics, "No Dice coefficient calculated"
            assert "iou" in metrics, "No IoU calculated"
            assert "precision" in metrics, "No precision calculated"
            assert "recall" in metrics, "No recall calculated"
            
            # Check reasonable metric values (these thresholds are placeholders)
            assert metrics["dice"] >= 0, "Dice coefficient should be non-negative"
            assert metrics["iou"] >= 0, "IoU should be non-negative"
            assert 0 <= metrics["precision"] <= 1, "Precision should be between 0 and 1"
            assert 0 <= metrics["recall"] <= 1, "Recall should be between 0 and 1"

@pytest.fixture
def drsam_validator(test_root):
    """Fixture to provide a DrSAMValidator instance."""
    dataset_path = test_root / "original_drsam"
    if not dataset_path.exists():
        pytest.skip("Original Dr-SAM dataset not found")
    return DrSAMValidator(dataset_path) 