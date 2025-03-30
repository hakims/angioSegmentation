# File: utils/io.py
# Purpose: Shared utilities for loading/saving media, frames, masks, boxes

import os
import cv2
import subprocess
from pathlib import Path
import imageio_ffmpeg
import numpy as np


FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov']

def detect_crop_bounds_from_video(video_path, threshold=10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video for crop detection: {video_path}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h

def crop_video_with_ffmpeg(input_path, output_path, crop_bounds):
    x, y, w, h = crop_bounds
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(input_path),
        "-filter:v", f"crop={w}:{h}:{x}:{y}",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to crop video: {input_path}")
        return False

def find_media_files(root_dir):
    """
    Find media files (images and videos) in a directory structure.
    Handles both nested folders and flat directories.
    
    Args:
        root_dir: Root directory to search
        
    Returns:
        Dictionary mapping folders to lists of media files
    """
    try:
        # Clean up the path - remove any quotes and normalize
        root_dir = str(root_dir).strip('"').strip("'")
        root_dir = Path(root_dir).resolve()
        
        # Debug logging
        print(f"\n🔍 Searching for media in: {root_dir}")
        print(f"📁 Directory exists: {root_dir.exists()}")
        print(f"📁 Is directory: {root_dir.is_dir()}")
        
        if not root_dir.exists():
            print(f"❌ Error: Directory does not exist: {root_dir}")
            return {}
            
        if not root_dir.is_dir():
            print(f"❌ Error: Path is not a directory: {root_dir}")
            return {}
            
        media_dict = {}
        
        # List all contents at root level
        print("\n📂 Root directory contents:")
        try:
            for item in root_dir.iterdir():
                print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
        except Exception as e:
            print(f"  ⚠️ Error listing directory contents: {str(e)}")
        
        # First, look for media files in nested directory structure
        print("\n🔍 Searching nested directories...")
        for root, dirs, files in os.walk(str(root_dir)):
            current_path = Path(root)
            print(f"\nChecking directory: {current_path}")
            
            media_files = []
            for file in files:
                try:
                    file_path = current_path / file
                    ext = file_path.suffix.lower()
                    if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                        print(f"  ✓ Found media file: {file}")
                        media_files.append(file_path)
                    else:
                        print(f"  ✗ Skipping non-media file: {file} (ext: {ext})")
                except Exception as e:
                    print(f"  ⚠️ Error processing file {file}: {str(e)}")
                    continue
            
            if media_files:
                media_dict[current_path] = media_files
                print(f"  ✅ Added {len(media_files)} files from {current_path}")
        
        # If no media found in nested structure, check flat directory
        if not media_dict:
            print("\n🔍 Checking for files in flat directory structure...")
            direct_files = []
            for ext in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS:
                try:
                    found_files = list(root_dir.glob(f"*{ext}"))
                    if found_files:
                        print(f"  ✓ Found {len(found_files)} files with extension {ext}")
                        direct_files.extend(found_files)
                except Exception as e:
                    print(f"  ⚠️ Error searching for {ext} files: {str(e)}")
            
            if direct_files:
                media_dict[root_dir] = direct_files
                print(f"  ✅ Added {len(direct_files)} files from flat directory")
        
        # Summary
        if not media_dict:
            print("\n❌ No media files found!")
            print(f"Supported extensions: {IMAGE_EXTENSIONS + VIDEO_EXTENSIONS}")
        else:
            print("\n✅ Media files found:")
            for folder, files in media_dict.items():
                print(f"\nFolder: {folder}")
                for f in files:
                    print(f"  - {f.name}")
        
        return media_dict
        
    except Exception as e:
        print(f"\n❌ Error while searching for media files:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Directory attempted: {root_dir}")
        return {}

def extract_frames_from_video(video_path, output_dir, use_fps=True, quiet=True):
    """
    Extract frames from video using ffmpeg. Returns list of paths to saved PNG frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = video_path.stem
    frame_rate = 2 if use_fps else 1
    output_pattern = str(output_dir / f"{base_name}-%04d.png")

    cmd = [
        FFMPEG_PATH, "-i", str(video_path),
        "-vf", f"fps={frame_rate}",
        output_pattern
    ]
    if quiet:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=True)

    return sorted(output_dir.glob(f"{base_name}-*.png"))

def get_frames_from_path(file_path, output_dir, use_fps=True, quiet=True, force_extract=False):
    """
    Get frames from a file path, handling both images and videos.
    For videos, extracts frames if needed or reuses existing ones.
    
    Args:
        file_path: Path to the file (image or video)
        output_dir: Directory to save extracted frames
        use_fps: Whether to use 2 fps (True) or 1 fps (False) for video extraction
        quiet: Whether to suppress ffmpeg output
        force_extract: Whether to force re-extraction of frames even if they exist
        
    Returns:
        List of frame paths (for image: list with single path, for video: list of extracted frames)
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    # If it's an image, just return it as a single-item list
    if ext in IMAGE_EXTENSIONS:
        return [file_path]
    
    # If it's a video, check if frames have already been extracted
    if ext in VIDEO_EXTENSIONS:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = file_path.stem
        
        # Check if frames already exist
        existing_frames = sorted(output_dir.glob(f"{base_name}-*.png"))
        
        if existing_frames and not force_extract:
            print(f"  📋 Using {len(existing_frames)} existing frames for {base_name}")
            return existing_frames
        
        # Need to extract frames - first check for cropping
        crop_bounds = detect_crop_bounds_from_video(file_path)
        if crop_bounds:
            # Create cropped directory at the same level as frames directory
            cropped_dir = output_dir.parent / "cropped"
            cropped_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cropped video in the cropped directory
            cropped_path = cropped_dir / f"{base_name}_cropped.mp4"
            
            # Check if cropped video already exists
            if cropped_path.exists() and not force_extract:
                print(f"  📋 Using existing cropped video: {cropped_path.name}")
                file_path = cropped_path
            elif crop_video_with_ffmpeg(file_path, cropped_path, crop_bounds):
                print(f"  ✂️ Saved cropped video: {cropped_path.name}")
                file_path = cropped_path
            else:
                print(f"  ⚠️ Cropping failed, using original video")
        
        # Extract frames
        return extract_frames_from_video(
            file_path,
            output_dir=output_dir,
            use_fps=use_fps,
            quiet=quiet
        )
    
    # If it's neither an image nor video, return empty list
    print(f"⚠️ Unsupported file format: {file_path}")
    return []
