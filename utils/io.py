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
    media_dict = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = Path(root) / file
            ext = file_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
                fin_folder = Path(root)
                if fin_folder not in media_dict:
                    media_dict[fin_folder] = []
                media_dict[fin_folder].append(file_path)
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
