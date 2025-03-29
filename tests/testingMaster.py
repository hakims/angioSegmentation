#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dr-SAM Testing Master
---------------------
Main entry point for all Dr-SAM testing.

This script provides a modular testing framework that can run different
test types and will be expanded in the future.

Usage:
    python tests/testingMaster.py [test_type] [options]

Test types:
    drsam           Run the original Dr-SAM validation test (default)
    performance     Run performance benchmarks (future)
    integration     Run integration tests (future)

Examples:
    # Run validation with default settings
    python tests/testingMaster.py

    # Run validation with specific method and visualization
    python tests/testingMaster.py drsam --method frangi --verbose

    # Only evaluate existing results
    python tests/testingMaster.py drsam --skip-pipeline
"""

import sys
import os
import subprocess
import argparse
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import validation utilities from the new location
from tests.originalDrSAM.dr_sam_validator import DrSAMValidator
from tests.originalDrSAM.drsam_testing_utils import run_drsam_pipeline


def run_drsam_validation(args):
    """
    Run validation against the original Dr-SAM dataset.
    
    This test:
    1. Loads original Dr-SAM dataset images from dr_sam/dataset/images
    2. Extracts expert bounding boxes from metadata.json
    3. Runs the buildDrSAM.py pipeline with these inputs
    4. Evaluates results by comparing to reference masks
    """
    print("\n=== Running Dr-SAM Validation Test ===")
    
    # Get the absolute path to the project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Setup paths using the project root for absolute paths
    dataset_path = project_root / Path(args.dataset)
    images_path = dataset_path / "images"
    metadata_path = dataset_path / "metadata.json"
    output_dir = project_root / Path(args.output_dir)
    
    print(f"Looking for dataset at: {dataset_path}")
    print(f"Looking for images at: {images_path}")
    print(f"Looking for metadata at: {metadata_path}")
    
    # Verify dataset structure
    if not images_path.exists():
        print(f"❌ Error: Images directory not found at {images_path}")
        return False
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata file not found at {metadata_path}")
        return False
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validator with appropriate settings
    validator = DrSAMValidator(
        dataset_path=dataset_path,
        output_dir=output_dir,
        visualize=args.visualize
    )
    
    # Try to add sample_size if it's in the validator's parameters
    if hasattr(validator, 'set_sample_size') and args.sample is not None:
        validator.set_sample_size(args.sample)
    
    # For evaluation-only mode, skip to results
    if args.skip_pipeline:
        print("\n=== Evaluating Existing Results ===")
        if hasattr(validator, 'evaluate_results'):
            results = validator.evaluate_results()
            return results is not None
        else:
            print("❌ Error: Validator does not have evaluate_results method")
            return False
    
    # Load expert bounding boxes - adjust attribute names based on DrSAMValidator
    try:
        # Check if the validator has image_boxes or prepare_expert_boundingBoxes method
        if hasattr(validator, 'image_boxes'):
            expert_boundingBoxes = validator.image_boxes
        elif hasattr(validator, 'prepare_expert_boundingBoxes'):
            # Get test images first
            test_images = validator.get_test_images(limit=args.sample)
            expert_boundingBoxes = validator.prepare_expert_boundingBoxes(test_images)
        else:
            # Try to load boundingBoxes directly from metadata file
            # This part is now handled within the DrSAMValidator
            expert_boundingBoxes = {}
        
        if not expert_boundingBoxes:
            print("⚠️ No expert bounding boxes were found in the metadata. Will proceed with auto-generation.")
            boundingBoxes_dict = None
        else:
            print(f"✓ Loaded {len(expert_boundingBoxes)} expert bounding boxes from metadata")
            boundingBoxes_dict = expert_boundingBoxes
    except Exception as e:
        print(f"❌ Error loading expert bounding boxes: {e}")
        print("⚠️ Will proceed with auto-generation.")
        boundingBoxes_dict = None
    
    # Use custom bounding boxes file if provided
    if args.boundingBoxes_file:
        try:
            boundingBoxes_file_path = project_root / Path(args.boundingBoxes_file)
            with open(boundingBoxes_file_path, 'r') as f:
                boundingBoxes_dict = json.load(f)
            print(f"✓ Loaded {len(boundingBoxes_dict)} bounding boxes from {args.boundingBoxes_file}")
        except Exception as e:
            print(f"❌ Error loading bounding boxes file: {e}")
            if expert_boundingBoxes:
                print("Falling back to expert bounding boxes")
                boundingBoxes_dict = expert_boundingBoxes
            else:
                print("No expert bounding boxes available. Will proceed with auto-generation.")
                boundingBoxes_dict = None
    elif not boundingBoxes_dict:
        print("No bounding boxes provided or found in metadata. Will use auto-generation.")
        # Leave boundingBoxes_dict as None to enable auto-generation
    
    # Run the pipeline
    try:
        print("\n=== Running Pipeline ===")
        
        # Auto bounding box mode if no bounding boxes provided
        auto_boundingBox = boundingBoxes_dict is None
        if auto_boundingBox:
            print("Using auto bounding box generation since no metadata boxes were found")
        
        success = run_drsam_pipeline(
            dataset_path=str(images_path),
            method=args.method,
            output_dir=output_dir,
            boundingBoxes_dict=boundingBoxes_dict,
            verbose=args.verbose
        )
        
        if not success:
            print("❌ Pipeline execution failed")
            return False
            
        print("✓ Pipeline executed successfully")
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False
    
    # Evaluate results
    if not args.skip_evaluation:
        print("\n=== Evaluating Results ===")
        # Check which method to use for evaluation
        if hasattr(validator, 'evaluate_results'):
            results = validator.evaluate_results()
        elif hasattr(validator, 'run_validation'):
            # Use the more generic run_validation method with appropriate parameters
            results = validator.run_validation(
                num_images=args.sample if args.sample else 5,
                method=args.method,
                run_pipeline=False,  # We already ran the pipeline
                save_results=True
            )
        else:
            print("❌ Error: No evaluation method found in validator")
            return False
        
        if results:
            return True
        else:
            print("❌ Evaluation failed")
            return False
    
    return True


def run_performance_benchmark(args):
    """Run performance benchmarks (placeholder for future implementation)."""
    print("\n=== Running Performance Benchmark ===")
    print("Performance benchmarks not yet implemented")
    return False


def run_integration_tests(args):
    """Run integration tests (placeholder for future implementation)."""
    print("\n=== Running Integration Tests ===")
    print("Integration tests not yet implemented")
    return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dr-SAM Testing Master - Main entry point for all tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run validation with default settings
  python tests/testingMaster.py
  
  # Run validation with specific method and visualization
  python tests/testingMaster.py drsam --method frangi --verbose
  
  # Only evaluate existing results
  python tests/testingMaster.py drsam --skip-pipeline
        """
    )
    
    # Main test type argument
    parser.add_argument("test_type", nargs="?", default="drsam",
                        choices=["drsam", "performance", "integration"],
                        help="Type of test to run (default: drsam)")
    
    # Common arguments
    parser.add_argument("--dataset", type=str, default="dr_sam/dataset",
                       help="Path to dataset directory (default: dr_sam/dataset)")
    parser.add_argument("--output-dir", type=str, default="tests/outputs",
                       help="Output directory for test results (default: tests/outputs)")
    parser.add_argument("--method", type=str, default="median",
                       help="Image transformation method to use (default: median)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations of results")
    
    # Validation-specific arguments
    parser.add_argument("--sample", type=int, default=None,
                       help="Number of images to sample (default: all)")
    parser.add_argument("--skip-pipeline", action="store_true",
                       help="Skip pipeline execution and only evaluate existing results")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation of results")
    parser.add_argument("--boundingBoxes-file", type=str, default=None,
                       help="Path to user-provided boundingBoxes JSON file")
    
    return parser.parse_args()


def main():
    """Entry point for testing framework."""
    args = parse_arguments()
    
    # Run appropriate test based on test_type
    if args.test_type == "drsam":
        success = run_drsam_validation(args)
    elif args.test_type == "performance":
        success = run_performance_benchmark(args)
    elif args.test_type == "integration":
        success = run_integration_tests(args)
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1
    
    # Return appropriate exit code
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 