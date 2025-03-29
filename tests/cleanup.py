#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cleanup script for removing empty test directories.
"""

import os
import shutil
import traceback
from pathlib import Path

def remove_directory(directory):
    """Remove a directory and print status."""
    try:
        if os.path.exists(directory):
            # Check if the directory is empty
            if not os.listdir(directory):
                try:
                    os.rmdir(directory)
                    print(f"✓ Removed empty directory: {directory}")
                except Exception as e:
                    print(f"❌ Error removing directory {directory} with os.rmdir: {e}")
                    print("Trying with shutil.rmtree...")
                    shutil.rmtree(directory)
                    print(f"✓ Removed directory with shutil.rmtree: {directory}")
            else:
                print(f"⚠️ Directory not empty: {directory}")
                print(f"   Contents: {os.listdir(directory)}")
        else:
            print(f"⚠️ Directory doesn't exist: {directory}")
    except Exception as e:
        print(f"❌ Error removing {directory}: {e}")
        traceback.print_exc()

def main():
    """Remove unused directories from tests folder."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    print(f"Script directory: {script_dir}")
    
    # Define directories to remove
    dirs_to_remove = [
        "unit",
        "integration", 
        "end_to_end",
        "resources"
    ]
    
    print("\nCleaning up unused test directories...")
    for dir_name in dirs_to_remove:
        directory = script_dir / dir_name
        print(f"\nProcessing: {directory}")
        remove_directory(directory)
    
    print("\nVerifying directories after cleanup...")
    for dir_name in dirs_to_remove:
        directory = script_dir / dir_name
        if os.path.exists(directory):
            print(f"❌ Directory still exists: {directory}")
        else:
            print(f"✓ Directory successfully removed: {directory}")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    main() 