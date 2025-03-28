import torch
import numpy as np
import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from skimage.filters import frangi, hessian
from skimage import img_as_ubyte

SAM_TYPE = "vit_h"
SAM_CHECKPOINT = str(Path(__file__).parent / "sam_vit_h.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_tophat(image, kernel_size=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)

def apply_frangi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = frangi(gray / 255.0)
    return cv2.cvtColor(img_as_ubyte(filtered), cv2.COLOR_GRAY2BGR)

def apply_hessian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = hessian(gray / 255.0)
    return cv2.cvtColor(img_as_ubyte(filtered), cv2.COLOR_GRAY2BGR)

def get_positive_points(image):
    """
    NOTE: This method assumes iodine contrast. 
    For CO₂ angiograms (vessels appear dark), a different logic is needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vessel_prob = 1.0 - gray.astype(np.float32) / 255.0
    mask = vessel_prob > 0.6

    border = 20
    h, w = mask.shape
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False

    points = np.column_stack(np.where(mask))
    if points.size == 0:
        return np.array([[w // 2, h // 2]]), np.array([1])
    sampled = points[np.random.choice(points.shape[0], size=min(5, len(points)), replace=False)]
    return sampled[:, [1, 0]], np.array([1] * sampled.shape[0])

def segment_vessels(image_path, transform="clahe"):
    image = cv2.imread(str(image_path))

    if transform == "clahe":
        image = apply_clahe(image)
    elif transform == "tophat":
        image = apply_tophat(image)
    elif transform == "frangi":
        image = apply_frangi(image)
    elif transform == "hessian":
        image = apply_hessian(image)
    # if 'none', do nothing

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    input_points, input_labels = get_positive_points(image)

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    return masks[0].astype(np.uint8)
