#!/usr/bin/env python
# File: tests/test_drsam_validation.py
# Purpose: Validate our implementation against the original Dr-SAM dataset

import os
import sys
import json
import pytest
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import validation utilities
from tests.test_utils.drsam_testing_utils import DrSAMValidator, run_drsam_pipeline


@pytest.fixture
def dataset_path(test_data_root):
    """Path to the original Dr-SAM dataset."""
    return test_data_root / "original_drsam"


@pytest.fixture
def output_dir(test_root):
    """Output directory for test results."""
    path = test_root / "test_outputs"
    path.mkdir(exist_ok=True, parents=True)
    return path


@pytest.mark.drsam
def test_drsam_replication(dataset_path, output_dir, request):
    """
    Validate our implementation against the original Dr-SAM dataset.
    
    This test:
    1. Loads original Dr-SAM dataset images
    2. Extracts expert bounding boxes from metadata
    3. Runs our pipeline with these inputs
    4. Evaluates results by comparing to reference masks
    
    Command line options:
    --method: Image transformation method (default: median)
    --visualize: Generate visualizations
    --sample: Number of images to test
    --skip-pipeline: Skip running the pipeline, only evaluate
    --skip-evaluation: Skip evaluation, only run pipeline
    """
    # Get command line options
    method = request.config.getoption("--method", default="median")
    visualize = request.config.getoption("--visualize", default=False)
    sample = request.config.getoption("--sample", default=None)
    skip_pipeline = request.config.getoption("--skip-pipeline", default=False)
    skip_evaluation = request.config.getoption("--skip-evaluation", default=False)
    
    # Set up paths
    images_path = dataset_path / "images"
    metadata_path = dataset_path / "metadata.json"
    
    # Verify dataset structure
    assert images_path.exists(), f"Images directory not found at {images_path}"
    assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"
    
    # Create validator
    validator = DrSAMValidator(
        dataset_path=dataset_path,
        output_dir=output_dir,
        visualize=visualize
    )
    
    # Set sample size if specified
    if sample is not None:
        validator.set_sample_size(int(sample))
    
    # For evaluation-only mode, skip pipeline execution
    if skip_pipeline:
        if hasattr(validator, 'evaluate_results'):
            results = validator.evaluate_results()
            assert results is not None, "Evaluation failed"
            return
        else:
            pytest.fail("Validator does not have evaluate_results method")
    
    # Get expert bounding boxes from metadata
    expert_boundingBoxes = validator.image_boxes if hasattr(validator, 'image_boxes') else None
    
    # Run the pipeline if not skipping
    images_path_str = str(images_path)
    run_success = run_drsam_pipeline(
        dataset_path=images_path_str,
        method=method,
        output_dir=output_dir,
        boundingBoxes_dict=expert_boundingBoxes,
        verbose=True
    )
    
    assert run_success, "Pipeline execution failed"
    
    # Skip evaluation if requested
    if skip_evaluation:
        return
    
    # Evaluate results
    if hasattr(validator, 'evaluate_results'):
        results = validator.evaluate_results()
    elif hasattr(validator, 'run_validation'):
        # Use the more generic run_validation method
        results = validator.run_validation(
            num_images=sample if sample else 5,
            method=method,
            run_pipeline=False,  # We already ran the pipeline
            save_results=True
        )
    else:
        pytest.fail("No evaluation method found in validator")
    
    assert results, "Evaluation failed"


def pytest_addoption(parser):
    """Add command-line options for the Dr-SAM validation test."""
    parser.addoption("--method", action="store", default="median",
                    help="Image transformation method to use")
    parser.addoption("--visualize", action="store_true",
                    help="Generate visualizations of results")
    parser.addoption("--sample", action="store", type=int, default=None,
                    help="Number of images to sample for testing")
    parser.addoption("--skip-pipeline", action="store_true",
                    help="Skip running the pipeline and evaluate existing results")
    parser.addoption("--skip-evaluation", action="store_true",
                    help="Skip evaluating results after running the pipeline") 