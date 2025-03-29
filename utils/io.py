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
    root_dir = Path(root_dir)
    media_dict = {}
    
    # First, look for media files in nested directory structure
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                fin_folder = Path(root)
                if fin_folder not in media_dict:
                    media_dict[fin_folder] = []
                media_dict[fin_folder].append(file_path)
    
    # If no media found in nested structure, check if this is a flat directory with media files
    if not media_dict:
        direct_files = []
        for ext in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS:
            direct_files.extend(list(root_dir.glob(f"*{ext}")))
        
        if direct_files:
            print(f"Found {len(direct_files)} media files directly in {root_dir}")
            media_dict[root_dir] = direct_files
    
    return media_dict

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
            cropped_path = file_path.with_name(file_path.stem + "_cropped.mp4")
            if crop_video_with_ffmpeg(file_path, cropped_path, crop_bounds):
                file_path = cropped_path
        
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
