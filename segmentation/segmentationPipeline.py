# File: segmentation/segmentationPipeline.py
# Version: 0.12 (standardizes terminology to use "boundingBox" consistently)
# Purpose: Run Dr-SAM on a list of frames and save all relevant outputs

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import traceback
from segmentation.segment_vessels import segment_vessels

def segment_and_save_outputs(frames, output_root, method="median", frame_to_boundingBoxes=None, frame_to_attributes=None):
    """
    Run Dr-SAM segmentation on each frame and save mask, skeletons, debug images.
    
    Args:
        frames: List of frames to process
        output_root: Root directory for outputs
        method: Image transformation method
        frame_to_boundingBoxes: Dictionary mapping filenames to bounding boxes
        frame_to_attributes: Dictionary mapping filenames to annotation attributes
    """
    frame_to_boundingBoxes = frame_to_boundingBoxes or {}
    frame_to_attributes = frame_to_attributes or {}

    mask_dir = output_root / "masks"
    debug_dir = output_root / "debug"
    raw_skel_dir = output_root / "skeletons_raw"
    pruned_skel_dir = output_root / "skeletons"
    meta_dir = output_root / "metadata"

    for d in [mask_dir, debug_dir, raw_skel_dir, pruned_skel_dir, meta_dir]:
        d.mkdir(exist_ok=True)

    for frame in frames:
        try:
            input_boundingBoxes = frame_to_boundingBoxes.get(frame.name)
            input_attributes = frame_to_attributes.get(frame.name, {})
            
            result = segment_vessels(str(frame), apply_anomaly_filter=True, method=method, input_boundingBoxes=input_boundingBoxes)

            suffix = f"_{method}" if method != "none" else ""

            for i, mask in enumerate(result["masks"]):
                out_name = f"{frame.stem}_mask_{i}.png"

                # Save mask
                mask_out = (1 - mask) * 255
                cv2.imwrite(str(mask_dir / out_name), mask_out.astype("uint8"))

                # Save skeletons
                if i < len(result["skeleton_raw"]):
                    raw_skel = result["skeleton_raw"][i]
                    cv2.imwrite(str(raw_skel_dir / out_name), raw_skel * 255)
                if i < len(result["skeleton"]):
                    skel = result["skeleton"][i]
                    cv2.imwrite(str(pruned_skel_dir / out_name), skel * 255)

                # Save debug overlay (original image)
                debug_img = cv2.imread(str(frame))
                cv2.imwrite(str(debug_dir / out_name), debug_img)

                # Save metadata (e.g., anomalies and attributes)
                meta_path = meta_dir / f"{frame.stem}_mask_{i}.json"
                metadata = {"anomalies": []}
                
                # Add anomalies if available
                if i < len(result["anomalies"]):
                    metadata["anomalies"] = result["anomalies"][i]
                
                # Add attributes if available for this mask
                if i in input_attributes:
                    metadata["attributes"] = input_attributes[i]
                elif len(input_attributes) == 1 and 0 in input_attributes and i == 0:
                    # If there's only one set of attributes and this is the first mask, use those
                    metadata["attributes"] = input_attributes[0]
                
                # Write metadata
                with open(meta_path, "w") as f:
                    import json
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed to segment {frame.name}: {e}")
            traceback.print_exc()
