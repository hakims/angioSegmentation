import torch
import numpy as np
import platform
import cv2
import os
from pathlib import Path
from PIL import Image, ImageFilter
from segment_anything import sam_model_registry, SamPredictor

from dr_sam_core.segmentation.point_finders import (
    find_positive_points,
    find_additional_positive_points
)

DEVICE = None  # Set dynamically by load_model()

def load_model(model_type="vit_h", model_path=None, device=None):
    """
    Loads a SAM model and returns a predictor. Dynamically selects the best available device.
    Priority: CUDA > MPS (on macOS) > CPU
    """
    global DEVICE

    system = platform.system().lower()
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif system == "darwin":
            try:
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    device = "mps"
                else:
                    device = "cpu"
            except AttributeError:
                device = "cpu"
        else:
            device = "cpu"

    DEVICE = device
    print(f"[INFO] Selected device: {DEVICE}")

    if model_path is None:
        model_path = "dr_sam_core/checkpoints/sam_vit_h.pth"

    model = sam_model_registry[model_type](checkpoint=model_path)
    model.to(device=DEVICE)
    return SamPredictor(model)

def create_masks(mask: torch.Tensor, bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    Creates a tensor of grayscale extracted regions for each bounding box in an image.
    Each region contains pixel values from the bounding box in the original image,
    with zeros elsewhere.
    """
    image_np = mask
    height, width = image_np.shape

    image_tensor = torch.from_numpy(image_np).float().to(device=DEVICE)
    regions = []

    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        region = torch.zeros((height, width), device=DEVICE)
        region[y_min:y_max, x_min:x_max] = image_tensor[y_min:y_max, x_min:x_max]
        regions.append(region)

    return torch.stack(regions)

def draw_points_on_image(image_np, points, radius=5, color=(0, 0, 255)):
    image_copy = image_np.copy()
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(image_copy, (x, y), radius, color, -1)
    return image_copy

def segmentize(image: np.ndarray, input_boxes: torch.Tensor, predictor, i: int = 3, image_path: str = None) -> dict:
    """
    Segments the image using the full Dr-SAM algorithm with iterative positive point refinement.
    
    This implementation follows the original Dr-SAM methodology by:
    1. Finding the best first point in each bounding box (darkest points with highest vessel density)
    2. Finding additional points to complete initial prompt set
    3. Pairing points for vessel segmentation by using the special [[0,3], [1,4], [2,5]] arrangement for 3-box case
    4. Running iterative refinement to improve segmentation quality
    5. Applying double ModeFilter to clean mask boundaries
    
    Args:
        image (np.ndarray): Input RGB image to segment
        input_boxes (torch.Tensor): Tensor of bounding boxes with shape [N, 4] where each box is [x1, y1, x2, y2]
        predictor (SamPredictor): Initialized SAM predictor object
        i (int): Number of refinement iterations (default: 3)
        image_path (str, optional): Path to save debug visualization
        
    Returns:
        dict: Dictionary containing:
            - "masks": Tensor of masks with shape [N, 1, H, W]
            - "input_points": List of points used for prompting the model
    """
    def _predict(image, input_boxes, input_points):
        predictor.set_image(image)
        masks = []
        
        # Verify input shapes match
        if len(input_boxes) != len(input_points):
            print(f"[WARNING] Shape mismatch: {len(input_boxes)} boxes but {len(input_points)} point groups")
            # Try to continue with smaller number
            min_len = min(len(input_boxes), len(input_points))
            input_boxes = input_boxes[:min_len]
            input_points = input_points[:min_len]
        
        for box_idx, (box, points) in enumerate(zip(input_boxes, input_points)):
            try:
                if len(points) == 0:
                    print(f"[WARNING] No points for box {box_idx}, skipping")
                    # Add empty mask of correct shape
                    h, w = image.shape[:2]
                    masks.append(np.zeros((1, h, w)))
                    continue
                    
                mask, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=np.ones(len(points)),
                    box=box.cpu().numpy(),
                    multimask_output=False,
                )
                masks.append(mask)
            except Exception as e:
                print(f"[ERROR] Prediction failed for box {box_idx}: {e}")
                # Add empty mask of correct shape as fallback
                h, w = image.shape[:2]
                masks.append(np.zeros((1, h, w)))

        if not masks:
            print("[WARNING] No masks were generated")
            h, w = image.shape[:2]
            return torch.zeros((0, 1, h, w), device=DEVICE)
        
        return torch.from_numpy(np.array(masks)).to(DEVICE)

    predicted = None

    input_points, input_label = find_positive_points(image, input_boxes)
    print(f"[DEBUG] Initial points: {input_points}")
    
    input_points, input_label = find_additional_positive_points(
        image, input_boxes, input_points, input_label
    )
    print(f"[DEBUG] After additional points: {input_points}")

    # Create paired points based on the number of boxes
    input_points_v2 = []
    num_boxes = len(input_boxes)
    
    # Use original DrSAM point pairing when exactly 3 boxes are present and 6+ points were found
    if num_boxes == 3 and len(input_points) >= 6:
        # Original DrSAM implementation paired points as [[0,3], [1,4], [2,5]]
        input_points_v2 = [[input_points[i], input_points[i + 3]] for i in range(3)]
        print("[INFO] Using original DrSAM point pairing for 3 boxes")
        print(f"[DEBUG] Original pairing: point 0 + point 3, point 1 + point 4, point 2 + point 5")
    else:
        # More flexible approach for different numbers of boxes
        for i_box in range(num_boxes):
            pts = []
            if i_box < len(input_points):
                pts.append(input_points[i_box])
            if (i_box + num_boxes) < len(input_points):
                pts.append(input_points[i_box + num_boxes])
            input_points_v2.append(pts)
        print(f"[INFO] Using flexible point pairing for {num_boxes} boxes")
        print(f"[DEBUG] Flexible pairing structure: {[[i, i+num_boxes] for i in range(num_boxes) if i+num_boxes < len(input_points)]}")

    for _ in range(i):
        predicted = _predict(image, input_boxes, np.array(input_points_v2))
        
        # For 3-box case, use the original DrSAM refinement logic
        if num_boxes == 3 and len(input_points_v2) == 3:
            for j in range(0, 3, 1):
                mask_points_v2 = []
                mask1 = np.transpose(predicted[j].cpu().numpy(), (1, 2, 0))
                
                im_pil = Image.fromarray(np.squeeze(mask1))
                im_pil = im_pil.filter(ImageFilter.ModeFilter(size=7))
                fixed_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))
                
                _img = image.copy()
                _img[fixed_mask.astype(np.bool8)] = [255, 255, 255]
                
                input_point_v2, input_label_v2 = find_positive_points(_img, [input_boxes[j]])
                mask_points_v2.extend(input_point_v2)
                input_points_v2[j].extend(mask_points_v2)
        # For other cases, use our flexible approach
        else:
            for j in range(len(input_boxes)):
                mask1 = np.transpose(predicted[j].cpu().numpy(), (1, 2, 0))
                im_pil = Image.fromarray(np.squeeze(mask1))
                im_pil = im_pil.filter(ImageFilter.ModeFilter(size=7))
                fixed_mask = np.asarray(im_pil.filter(ImageFilter.ModeFilter(size=7)))

                _img = image.copy()
                _img[fixed_mask.astype(np.bool8)] = [255, 255, 255]

                new_pts, _ = find_positive_points(_img, [input_boxes[j]])
                input_points_v2[j].extend(new_pts)

    print(f"[DEBUG] input_boxes: {input_boxes}")
    print(f"[DEBUG] input_points: {input_points_v2}")

    predicted_final = _predict(image, input_boxes, np.array(input_points_v2))
    print(f"[DEBUG] Final predicted mask shape: {predicted_final.shape}")
    if predicted_final.shape[1:] != (1, image.shape[0], image.shape[1]):
        print("[WARNING] Predicted mask shape unexpected.")

    # Check if the final mask is all zeros
    mask_array = predicted_final[0, 0].cpu().numpy()
    nonzero_count = np.count_nonzero(mask_array)
    print(f"[DEBUG] Non-zero pixels in final mask: {nonzero_count}")
    if nonzero_count == 0:
        print("[WARNING] Final mask is completely empty (all zeros)")

    # Draw and save prompt points debug image to debug folder if path is available
    if image_path:
        flat_points = [pt for sublist in input_points_v2 for pt in sublist]
        debug_img = draw_points_on_image(image, flat_points)
        image_path = Path(image_path)
        debug_dir = image_path.parent.parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        out_path = debug_dir / f"{image_path.stem}_debug_prompt_points.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    return {
        "masks": predicted_final,
        "input_points": input_points_v2.tolist() if isinstance(input_points_v2, np.ndarray) else input_points_v2
    }
