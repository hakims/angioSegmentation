#!/usr/bin/env python
# File: tests/test_utils/file_utils.py
# Purpose: File utilities for testing, building on utils/io.py but specialized for test needs

import os
from pathlib import Path
from typing import List, Dict, Optional

# Extend the image extensions list from utils/io.py to support more formats for testing
TEST_IMAGE_EXTENSIONS = [
    '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', 
    '.PNG', '.JPG', '.JPEG', '.TIF', '.TIFF', '.BMP'
]

def find_image_files(directory: Path) -> List[Path]:
    """
    Find all image files in a directory, supporting common image formats.
    
    This is an extended version of functionality in utils/io.py to support
    more image formats needed for testing.
    
    Args:
        directory: Directory to search for image files
        
    Returns:
        List of paths to image files, sorted by name
    """
    if not directory.exists() or not directory.is_dir():
        return []
        
    # Look for all supported image formats
    image_files = []
    for ext in TEST_IMAGE_EXTENSIONS:
        image_files.extend(list(directory.glob(f"*{ext}")))
    
    return sorted(image_files)

def find_all_image_files(root_dir: Path, recursive: bool = True) -> List[Path]:
    """
    Find all image files in a directory tree, recursively.
    
    Args:
        root_dir: Root directory to search
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of paths to image files, sorted by path
    """
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    
    image_files = []
    
    if recursive:
        # Recursive search through subdirectories
        for root, _, files in os.walk(root_dir):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                if any(file_path.name.lower().endswith(ext.lower()) for ext in TEST_IMAGE_EXTENSIONS):
                    image_files.append(file_path)
    else:
        # Non-recursive search (current directory only)
        for ext in TEST_IMAGE_EXTENSIONS:
            image_files.extend(list(root_dir.glob(f"*{ext}")))
    
    return sorted(image_files)

def find_images_by_pattern(directory: Path, pattern: str) -> List[Path]:
    """
    Find image files matching a specific pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match (e.g., "*frame*")
        
    Returns:
        List of matching image file paths
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    image_files = []
    for ext in TEST_IMAGE_EXTENSIONS:
        image_files.extend(list(directory.glob(f"{pattern}{ext}")))
    
    return sorted(image_files)

def group_images_by_directory(root_dir: Path) -> Dict[Path, List[Path]]:
    """
    Find all image files and group them by directory.
    
    This is similar to find_media_files() in utils/io.py but specialized for images.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        Dictionary mapping directories to lists of image files
    """
    root_dir = Path(root_dir)
    image_dict = {}
    
    # Look for image files in nested directory structure
    for root, _, files in os.walk(root_dir):
        dir_images = []
        root_path = Path(root)
        
        for file in files:
            file_path = root_path / file
            if any(file_path.name.lower().endswith(ext.lower()) for ext in TEST_IMAGE_EXTENSIONS):
                dir_images.append(file_path)
        
        if dir_images:
            image_dict[root_path] = sorted(dir_images)
    
    return image_dict 