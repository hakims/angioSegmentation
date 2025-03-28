# File: buildDrSAM.py
# Version: 0.15 (restores environment bootstrapping, torch, and masks_dir setup)
"""
Dr-SAM Angiogram Segmentation Pipeline
---------------------------------------
Adds support for image preprocessing filters prior to segmentation.

Run this script after installing dependencies with:
    python buildDrSAM.py <root_folder>
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

from utils.io import (
    detect_crop_bounds_from_video,
    crop_video_with_ffmpeg,
    find_media_files,
    extract_frames_from_video,
)
from segmentation.segmentationPipeline import segment_and_save_outputs
from preprocessing.bounding_boxes import (
    generate_boxes_from_image,
    overlay_debug_outputs,
)
from preprocessing.transforms import apply_transform

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# === Dynamic dependency and setup bootstrap ===
def _run_build_dependencies_if_needed():
    print("🔍 Checking environment dependencies...")
    try:
        import torch
        print("✅ PyTorch is installed.")
        print("🎉 All dependencies already installed.")
    except ImportError:
        print("⚙️ Running buildDependencies.py to set up environment...")
        subprocess.check_call([sys.executable, "buildDependencies.py"])
        print("✅ Finished environment setup via buildDependencies.py")

_run_build_dependencies_if_needed()

def main(mp4_root, use_fps, transform="none", quiet=True, auto_box=True, box_transform="frangi", min_box_size=2000, save_boxes=True):
    print(f"📂 Scanning: {mp4_root}")
    media_by_folder = find_media_files(mp4_root)
    total_start = time.time()

    for folder, files in media_by_folder.items():
        print(f"\n📁 Processing folder: {folder}")
        folder_boxes = {}

        debug_boxes_dir = folder / "debug" / "bounding_boxes"
        debug_boxes_dir.mkdir(parents=True, exist_ok=True)

        masks_dir = folder / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        for idx, file in enumerate(files, 1):
            ext = file.suffix.lower()
            if ext not in ['.mp4', '.avi', '.mov']:
                print(f"🧾 Skipping non-video file: {file.name}")
                continue

            print(f"  ▶ [{idx}/{len(files)}] {file.name}")
            start_time = time.time()

            crop_bounds = detect_crop_bounds_from_video(file)
            if crop_bounds:
                cropped_path = file.with_name(file.stem + "_cropped.mp4")
                if crop_video_with_ffmpeg(file, cropped_path, crop_bounds):
                    file = cropped_path

            frame_paths = extract_frames_from_video(
                file,
                output_dir=folder / "frames",
                use_fps=use_fps,
                quiet=quiet
            )

            for frame in frame_paths:
                image = cv2.imread(str(frame))
                if image is None:
                    print(f"⚠️ Skipping unreadable frame: {frame.name}")
                    continue

                boxes = generate_boxes_from_image(image, transform=box_transform, min_size=min_box_size)
                folder_boxes[frame.name] = boxes.tolist()

                debug_prefix = debug_boxes_dir / frame.stem
                vessel_map = image if transform == "none" else apply_transform(image, method=box_transform)
                overlay_debug_outputs(image, vessel_map, boxes, str(debug_prefix))

            # Segment and save for all frames in this folder
            segment_and_save_outputs(
                frames=frame_paths,
                output_root=folder,
                transform=transform,
                frame_to_boxes=folder_boxes
            )

            elapsed = time.time() - start_time
            minutes, seconds = divmod(elapsed, 60)
            hours, minutes = divmod(minutes, 60)
            if hours >= 1:
                print(f"    ⏱️ Done in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            elif minutes >= 1:
                print(f"    ⏱️ Done in {int(minutes)}m {seconds:.2f}s")
            else:
                print(f"    ⏱️ Done in {seconds:.2f}s")

        if auto_box and save_boxes:
            out_path = folder / "auto_boxes.json"
            with open(out_path, "w") as f:
                json.dump(folder_boxes, f, indent=2)
            print(f"📦 Saved auto-generated boxes to {out_path}")

    total_elapsed = time.time() - total_start
    minutes, seconds = divmod(total_elapsed, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        print(f"\n✅ Finished entire run in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif minutes >= 1:
        print(f"\n✅ Finished entire run in {int(minutes)}m {seconds:.2f}s")
    else:
        print(f"\n✅ Finished entire run in {seconds:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dr-SAM segmentation pipeline")
    parser.add_argument("root_dir", type=str, help="Path to angios folder")
    parser.add_argument("--use-fps", action="store_true", help="Use 2 fps frame stepping (default: 1 fps)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose ffmpeg output")
    parser.add_argument("--transform", type=str, default="none", help="Image transform: none, clahe, tophat, frangi, hessian")
    parser.add_argument("--box-transform", type=str, default="frangi", help="Transform to apply before box generation")
    parser.add_argument("--min-box-size", type=int, default=2000, help="Minimum size for keeping auto boxes")
    args = parser.parse_args()

    main(
        Path(args.root_dir),
        use_fps=args.use_fps,
        transform=args.transform,
        quiet=not args.verbose,
        auto_box=True,
        box_transform=args.box_transform,
        min_box_size=args.min_box_size
    )
