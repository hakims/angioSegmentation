#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared utilities for cleanup operations.
"""

import os
import shutil
import traceback
from pathlib import Path
from typing import Set, List, Optional

def remove_directory(directory: Path, force: bool = False) -> bool:
    """
    Remove a directory and print status.
    
    Args:
        directory: Path to directory to remove
        force: Whether to force remove using system commands
        
    Returns:
        bool: True if directory was removed, False otherwise
    """
    try:
        if not directory.exists():
            print(f"⚠️ Directory doesn't exist: {directory}")
            return False
            
        # Try normal removal first
        try:
            if not os.listdir(directory):
                os.rmdir(directory)
                print(f"✓ Removed empty directory: {directory}")
                return True
        except Exception as e:
            print(f"❌ Error removing directory {directory} with os.rmdir: {e}")
            
        # Try shutil.rmtree
        print("Trying with shutil.rmtree...")
        shutil.rmtree(directory, ignore_errors=True)
        if not directory.exists():
            print(f"✓ Removed directory with shutil.rmtree: {directory}")
            return True
            
        # If force is True and directory still exists, try system command
        if force and directory.exists():
            print("Trying with system command...")
            if os.name == 'nt':  # Windows
                os.system(f'rd /s /q "{directory}"')
            else:  # Unix/Linux
                os.system(f'rm -rf "{directory}"')
            if not directory.exists():
                print(f"✓ Removed directory with system command: {directory}")
                return True
                
        if directory.exists():
            print(f"❌ Failed to remove directory: {directory}")
            print(f"   Contents: {os.listdir(directory)}")
            return False
            
    except Exception as e:
        print(f"❌ Error removing {directory}: {e}")
        traceback.print_exc()
        return False

def cleanup_directory_contents(directory: Path, 
                            preserve_extensions: Optional[Set[str]] = None,
                            remove_patterns: Optional[List[str]] = None) -> tuple[int, int]:
    """
    Clean up contents of a directory while preserving specified files.
    
    Args:
        directory: Directory to clean
        preserve_extensions: Set of file extensions to preserve
        remove_patterns: List of patterns to match for removal
        
    Returns:
        tuple[int, int]: Number of files and directories removed
    """
    if preserve_extensions is None:
        preserve_extensions = set()
    if remove_patterns is None:
        remove_patterns = []
        
    files_removed = 0
    dirs_removed = 0
    
    try:
        for root, dirs, files in os.walk(directory, topdown=False):
            current_path = Path(root)
            
            # Remove files
            for file in files:
                file_path = current_path / file
                
                # Skip if file should be preserved
                if file_path.suffix.lower() in preserve_extensions:
                    continue
                    
                # Check if file matches any remove patterns
                should_remove = False
                for pattern in remove_patterns:
                    if file_path.name.endswith(pattern):
                        should_remove = True
                        break
                        
                if should_remove:
                    try:
                        file_path.unlink()
                        files_removed += 1
                        print(f"  🗑️ Deleted file: {file_path}")
                    except Exception as e:
                        print(f"    ❌ Error removing file: {e}")
                        
            # Remove directories
            for dir_name in dirs:
                dir_path = current_path / dir_name
                if remove_directory(dir_path):
                    dirs_removed += 1
                    
    except Exception as e:
        print(f"❌ Error cleaning directory {directory}: {e}")
        traceback.print_exc()
        
    return files_removed, dirs_removed 