# File: buildDrSAM.py
# Version: 0.28 (adds COCO format support)
"""
Dr-SAM Angiogram Segmentation Pipeline
---------------------------------------
Adds support for image preprocessing filters prior to segmentation.

Run this script after installing dependencies with:
    python buildDrSAM.py <root_folder>

For testing, use the dedicated testing framework:
    python tests/run_tests.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import subprocess
import os
import argparse
import time
import json
import cv2
import torch
import imageio_ffmpeg
import glob

from utils.io import (
    find_media_files,
    get_frames_from_path,
)
from segmentation.segmentationPipeline import segment_and_save_outputs
from preprocessing.bounding_boxes import process_images, load_bounding_boxes, save_bounding_boxes
from preprocessing.transforms import apply_transform

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# === Dynamic dependency and setup bootstrap ===
def _run_build_dependencies_if_needed():
    print("[INFO] Checking environment dependencies...")
    try:
        import torch
        print("[OK] PyTorch is installed.")
        print("[OK] All dependencies already installed.")
    except ImportError:
        print("[INFO] Running buildDependencies.py to set up environment...")
        subprocess.check_call([sys.executable, "buildDependencies.py"])
        print("[OK] Finished environment setup via buildDependencies.py")

_run_build_dependencies_if_needed()

def main(input_path, use_fps=False, method="median", quiet=True, auto_boundingBox=True, 
         boundingBox_method="frangi", min_boundingBox_size=2000, user_provided_boundingBoxes=None, 
         user_provided_attributes=None, boundingBox_file=None, output_dir=None, 
         force_extract=False, output_format="both", validate_schema=True):
    """
    Main DrSAM pipeline function.
    
    Args:
        input_path: Path to folder containing MP4 files or images
        use_fps: Whether to use 2 fps for frame extraction
        method: Image transformation method to use
        quiet: Suppress verbose output
        auto_boundingBox: Whether to auto-generate bounding boxes when no metadata is available
        boundingBox_method: Method for automatic bounding box generation
        min_boundingBox_size: Minimum size for keeping auto-generated bounding boxes
        user_provided_boundingBoxes: Optional dictionary mapping filenames to pre-defined bounding boxes 
        user_provided_attributes: Optional dictionary mapping filenames to annotation attributes
        boundingBox_file: Path to JSON file containing bounding box information
        output_dir: Custom output directory
        force_extract: Whether to force re-extraction of frames from videos
        output_format: Format for output annotations ('drsam', 'coco', or 'both')
        validate_schema: Whether to validate metadata against the CVAT labeling schema
    """
    print(f"�� Processing: {input_path}")
    input_path = Path(input_path)
    total_start = time.time()
    
    # Initialize boundingBoxes dictionary from user_provided_boundingBoxes if provided
    boundingBoxes_dict = user_provided_boundingBoxes.copy() if user_provided_boundingBoxes else {}
    
    # Initialize attributes dictionary from user_provided_attributes if provided
    attributes_dict = user_provided_attributes.copy() if user_provided_attributes else {}
    
    # If no pre-loaded boundingBoxes, try to load from file or search in input folder
    if not boundingBoxes_dict:
        boundingBoxes_dict, attr_dict = load_bounding_boxes(boundingBox_file, input_path)
        if attr_dict:
            attributes_dict.update(attr_dict)
    
    # Validate all attributes against schema if requested
    if validate_schema and attributes_dict:
        from utils.schema_utils import validate_and_fix_attributes_dict
        
        print(f"Validating {sum(len(attrs) for attrs in attributes_dict.values())} attribute sets against schema...")
        
        # Validate and fix attributes according to schema
        attributes_dict, validation_stats = validate_and_fix_attributes_dict(attributes_dict)
        
        # Report validation results
        if validation_stats["total_issues"] > 0:
            print(f"Schema validation fixed issues in {validation_stats['fixed_issues']} attribute sets")
        else:
            print("All attributes are schema-compliant!")
    
    # Determine if we should auto-generate bounding boxes
    should_auto_generate = auto_boundingBox and not boundingBoxes_dict
    
    # Set up output folder
    output_folder = Path(output_dir) if output_dir else input_path
    
    # Report bounding box status
    if boundingBoxes_dict:
        print(f"Using {len(boundingBoxes_dict)} bounding boxes from metadata")
    elif should_auto_generate:
        print(f"No bounding box metadata found. Will auto-generate using {boundingBox_method} filter.")
    else:
        print(f"Auto bounding box generation is disabled. Please provide a JSON file with bounding box data.")
    
    # Find media files - this will handle both nested and flat directory structures
    folders_to_process = find_media_files(input_path)
    
    if not folders_to_process:
        print("⚠️ No media files found to process!")
        return
    
    # Process each folder
    for folder, files in folders_to_process.items():
        # Determine whether to use the custom output folder or the media folder
        target_output = output_folder if output_dir else folder
        
        print(f"\n📁 Processing folder: {folder}")
        if folder != target_output:
            print(f"  📤 Output directory: {target_output}")
        
        # Create necessary directories
        masks_dir = target_output / "masks"
        masks_dir.mkdir(exist_ok=True, parents=True)
        
        # Process all files in the folder
        all_frames = []
        
        for idx, file in enumerate(files, 1):
            file_name = file.name
            print(f"  ▶ [{idx}/{len(files)}] {file_name}")
            start_time = time.time()
            
            # Get frames from the file (image or video)
            frame_paths = get_frames_from_path(
                file,
                output_dir=target_output / "frames",
                use_fps=use_fps,
                quiet=quiet,
                force_extract=force_extract
            )
            
            # Add frames to the collection
            all_frames.extend(frame_paths)
            
            process_time = time.time() - start_time
            print(f"  ✓ Processed in {process_time:.2f}s")
        
        # Generate bounding boxes and segment all frames
        if all_frames:
            # Process images to generate bounding boxes or use existing ones
            boundingBoxes_dict, attributes_dict = process_images(
                image_files=all_frames,
                output_folder=target_output,
                method=boundingBox_method,
                min_box_size=min_boundingBox_size,
                existing_boundingBoxes=boundingBoxes_dict,
                existing_attributes=attributes_dict,
                auto_generate=should_auto_generate,
                save_debug=True
            )
            
            # Save annotations in the requested format(s)
            if output_format != "both":
                # If only one format is requested, delete the other format that was created by process_images
                if output_format == "drsam":
                    coco_path = target_output / "annotations_coco.json"
                    if coco_path.exists():
                        os.remove(coco_path)
                elif output_format == "coco":
                    drsam_path = target_output / "boundingBoxes.json"
                    if drsam_path.exists():
                        os.remove(drsam_path)
            
            # Segment and save outputs for all frames
            segment_and_save_outputs(
                frames=all_frames,
                output_root=target_output,
                method=method,
                frame_to_boundingBoxes=boundingBoxes_dict,
                frame_to_attributes=attributes_dict
            )
    
    total_time = time.time() - total_start
    print(f"\n✨ Total processing time: {total_time:.2f}s")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dr-SAM Angiogram Segmentation Pipeline")
    
    # Main arguments
    parser.add_argument("input", nargs="?", default=None,
                       help="Path to input folder containing videos/images")
    parser.add_argument("--method", type=str, default="median", 
                       help="Transformation method (none, median, frangi, clahe, tophat, hessian)")
    parser.add_argument("--boundingBox-method", type=str, default="frangi",
                       choices=["frangi", "clahe", "hessian", "median", "original", "raw"],
                       help="Method to use for vessel map generation")
    parser.add_argument("--fps", action="store_true",
                       help="Extract frames at 2 FPS instead of 1 frame/sec")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    # Bounding box generation options
    parser.add_argument("--no-auto-boundingBox", dest="auto_boundingBox", action="store_false",
                       help="Disable automatic bounding box generation (requires bounding box metadata)")
    parser.add_argument("--min-boundingBox-size", type=int, default=2000,
                       help="Minimum size for auto-generated bounding boxes")
    
    # Processing options
    parser.add_argument("--force-extract", action="store_true",
                       help="Force re-extraction of frames from videos even if they exist")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Custom output directory")
    parser.add_argument("--user-provided-boundingBoxes", type=str, default=None,
                       help="Path to JSON file with bounding box metadata (Dr-SAM or COCO format)")
    parser.add_argument("--output-format", type=str, default="both", choices=["drsam", "coco", "both"],
                       help="Format for output annotations")
    
    # Testing arguments
    parser.add_argument("--validate-schema", action="store_true",
                       help="Validate metadata against the CVAT labeling schema")
    
    # Set defaults
    parser.set_defaults(auto_boundingBox=True)
    
    args = parser.parse_args()
    
    # If no input is provided, show help and exit
    if args.input is None:
        parser.print_help()
        sys.exit(1)
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    # Run main pipeline
    main(
        input_path=args.input,
        use_fps=args.fps,
        method=args.method,
        quiet=args.quiet,
        auto_boundingBox=args.auto_boundingBox,
        boundingBox_method=args.boundingBox_method,
        min_boundingBox_size=args.min_boundingBox_size,
        user_provided_boundingBoxes=None,  # No pre-loaded boxes in normal mode
        user_provided_attributes=None,
        boundingBox_file=args.user_provided_boundingBoxes,
        output_dir=args.output_dir,
        force_extract=args.force_extract,
        output_format=args.output_format,
        validate_schema=args.validate_schema
    )
