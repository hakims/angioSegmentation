#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cleanup script for removing test directories and outputs.
This script is specifically for cleaning up the test environment,
separate from the main cleanup.py which handles generated files.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.cleanup_utils import remove_directory

# Define paths to match conftest.py
TEST_OUTPUT_DIR = Path(__file__).parent / "outputs"
VISUALIZATION_DIR = TEST_OUTPUT_DIR / "visualizations"

def main():
    """Remove all test output directories and files."""
    parser = argparse.ArgumentParser(description="Clean up test outputs and visualizations")
    parser.add_argument("path", nargs="?", help="Path to clean up (defaults to tests/outputs)")
    parser.add_argument("--keep-visualizations", action="store_true", 
                       help="Keep visualization files while cleaning other test outputs")
    args = parser.parse_args()
    
    # Get the outputs directory
    if args.path:
        outputs_dir = Path(args.path).resolve()
    else:
        outputs_dir = TEST_OUTPUT_DIR
        
    print(f"Cleaning up test outputs in: {outputs_dir}")
    
    if not outputs_dir.exists():
        print("No test outputs directory found.")
        return
    
    if args.keep_visualizations:
        print("\nKeeping visualization files...")
        # Remove everything except visualizations directory
        for item in outputs_dir.iterdir():
            if item.is_dir() and item.name != "visualizations":
                print(f"Removing directory: {item}")
                remove_directory(item, force=True)
            elif item.is_file():
                print(f"Removing file: {item}")
                try:
                    item.unlink()
                except Exception as e:
                    print(f"Error removing file: {e}")
    else:
        # Remove the entire outputs directory and all its contents
        print("\nRemoving all test outputs...")
        remove_directory(outputs_dir, force=True)
    
    # Verify cleanup
    if args.keep_visualizations:
        # Check if only visualizations directory remains
        remaining_items = list(outputs_dir.iterdir())
        if len(remaining_items) == 1 and remaining_items[0].name == "visualizations":
            print("✓ Successfully cleaned up test outputs while preserving visualizations")
        else:
            print("❌ Some items could not be removed")
    else:
        if outputs_dir.exists():
            print("❌ Failed to remove outputs directory")
        else:
            print("✓ Successfully removed all test outputs")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    main() 