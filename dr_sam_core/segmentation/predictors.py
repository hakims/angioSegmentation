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

def segmentize(image: np.ndarray, input_boxes: torch.Tensor, predictor, i: int = 3, image_path: str = None) -> torch.Tensor:
    """
    Segments the image using the full Dr-SAM algorithm with iterative positive point refinement.
    """
    def _predict(image, input_boxes, input_points):
        predictor.set_image(image)
        masks = []
        for box, points in zip(input_boxes, input_points):
            mask, _, _ = predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                box=box.cpu().numpy(),
                multimask_output=False,
            )
            masks.append(mask)

        return torch.from_numpy(np.array(masks)).to(DEVICE)

    predicted = None

    input_points, input_label = find_positive_points(image, input_boxes)
    input_points, input_label = find_additional_positive_points(
        image, input_boxes, input_points, input_label
    )

    input_points_v2 = []
    num_boxes = len(input_boxes)
    for i_box in range(num_boxes):
        pts = []
        if i_box < len(input_points):
            pts.append(input_points[i_box])
        if (i_box + num_boxes) < len(input_points):
            pts.append(input_points[i_box + num_boxes])
        input_points_v2.append(pts)

    for _ in range(i):
        predicted = _predict(image, input_boxes, np.array(input_points_v2))
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

    return predicted_final
