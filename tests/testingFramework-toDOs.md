# Testing Suite Implementation Plan

## 1. Setup Testing Framework

1. **Create Test Directory Structure**
   - Create a `tests` directory at the root level
   - Add subdirectories for different test types:
     - `unit/` - For unit tests of individual components
     - `integration/` - For testing interactions between components
     - `end_to_end/` - For full pipeline tests
     - `resources/` - For test data and expected outputs

2. **Set Up Testing Dependencies**
   - Add `pytest` as the primary testing framework
   - Create a `requirements-dev.txt` file for development dependencies
   - Add image comparison tools like SSIM or MSE for mask comparison

## 2. Unit Tests Implementation

1. **Transform Tests**
   - Test each image transformation function independently
   - Verify correct output dimensions and types
   - Compare outputs against known reference images

2. **Point Finding Tests**
   - Test point finding functions against sample images with known points
   - Verify correct number and approximate locations of points

3. **Bounding Box Generation Tests**
   - Test automated bounding box generation
   - Compare against reference boxes for known images
   - Verify handling of edge cases (empty images, low contrast)

4. **Model Loading Tests**
   - Test SAM model loading with different parameters
   - Verify correct device selection logic
   - Test with mock models for speed

## 3. Integration Tests

1. **Segmentation Pipeline Tests**
   - Test combinations of transforms and segmentation
   - Verify correct chaining of operations
   - Test error handling and fallbacks

2. **IO Integration Tests**
   - Test frame extraction from videos
   - Test saving and loading of masks and metadata
   - Verify correct folder structure creation

## 4. End-to-End Validation Tests

1. **Original DrSAM Replication Test**
   - Create a test that uses the original DrSAM dataset
   - Run segmentation with expert-labeled bounding boxes
   - Compare outputs to reference masks using IoU or Dice coefficient
   - Success criterion: Achieve similar or better performance metrics

2. **Auto Box Generation Test**
   - Run the pipeline with auto-generated boxes
   - Compare against expert boxes and original DrSAM results
   - Measure and report performance differences

3. **Performance Benchmarking**
   - Test processing speed for different configurations
   - Measure memory usage
   - Establish baseline performance metrics

## 5. Continuous Integration Setup

1. **GitHub Actions Configuration**
   - Set up automated testing on pull requests
   - Configure test result reporting
   - Add coverage reporting

2. **Pre-commit Hooks**
   - Set up linting and code formatting
   - Add test running for critical components

## 6. Specific Implementation Details

### A. Original DrSAM Validation Test

```python
def test_original_drsam_replication():
    """Test that our implementation produces similar results to original DrSAM."""
    # Load test images and reference masks
    test_images = load_test_images("dr_sam/dataset/images")
    reference_masks = load_reference_masks("dr_sam/dataset/masks")
    metadata = load_metadata("dr_sam/dataset/metadata.json")
    
    results = {}
    for image_id, image in test_images.items():
        # Get expert bounding boxes from metadata
        bboxes = get_bboxes_for_image(metadata, image_id)
        
        # Run our implementation with default settings
        our_result = segment_vessels(
            image_path=image,
            method="median",  # Use the original DrSAM filter
            input_boxes=bboxes
        )
        
        # Compare our mask with reference mask
        reference_mask = reference_masks[image_id]
        iou_score = calculate_iou(our_result["masks"], reference_mask)
        dice_score = calculate_dice(our_result["masks"], reference_mask)
        
        # Store results
        results[image_id] = {
            "iou": iou_score,
            "dice": dice_score
        }
    
    # Calculate overall metrics
    avg_iou = np.mean([r["iou"] for r in results.values()])
    avg_dice = np.mean([r["dice"] for r in results.values()])
    
    # Assert acceptable performance
    assert avg_iou > 0.8, f"Average IoU ({avg_iou}) below threshold"
    assert avg_dice > 0.85, f"Average Dice ({avg_dice}) below threshold"
    
    # Save detailed results for reporting
    save_validation_results(results, "validation_results.json")
```

### B. Auto Box Generation Test

```python
def test_auto_box_generation():
    """Test our auto box generation against expert boxes."""
    # Load test images and reference masks
    test_images = load_test_images("dr_sam/dataset/images")
    reference_masks = load_reference_masks("dr_sam/dataset/masks")
    metadata = load_metadata("dr_sam/dataset/metadata.json")
    
    auto_vs_expert_results = {}
    for image_id, image in test_images.items():
        # Get expert bounding boxes
        expert_bboxes = get_bboxes_for_image(metadata, image_id)
        
        # Generate auto boxes
        image_array = cv2.imread(image)
        auto_bboxes = generate_boxes_from_image(
            image_array, 
            method="frangi",
            min_size=2000
        )
        
        # Run with expert boxes
        expert_result = segment_vessels(
            image_path=image,
            method="median",
            input_boxes=expert_bboxes
        )
        
        # Run with auto boxes
        auto_result = segment_vessels(
            image_path=image,
            method="median",
            input_boxes=auto_bboxes
        )
        
        # Compare results
        expert_iou = calculate_iou(expert_result["masks"], reference_masks[image_id])
        auto_iou = calculate_iou(auto_result["masks"], reference_masks[image_id])
        
        # Box overlap metric
        box_overlap = calculate_box_overlap(expert_bboxes, auto_bboxes)
        
        auto_vs_expert_results[image_id] = {
            "expert_iou": expert_iou,
            "auto_iou": auto_iou,
            "box_overlap": box_overlap,
            "expert_boxes": expert_bboxes.tolist(),
            "auto_boxes": auto_bboxes.tolist()
        }
    
    # Calculate overall metrics
    avg_expert_iou = np.mean([r["expert_iou"] for r in auto_vs_expert_results.values()])
    avg_auto_iou = np.mean([r["auto_iou"] for r in auto_vs_expert_results.values()])
    avg_box_overlap = np.mean([r["box_overlap"] for r in auto_vs_expert_results.values()])
    
    # Report results without hard assertions (this is exploratory)
    print(f"Expert boxes average IoU: {avg_expert_iou:.4f}")
    print(f"Auto boxes average IoU: {avg_auto_iou:.4f}")
    print(f"Average box overlap: {avg_box_overlap:.4f}")
    
    # Save detailed results for analysis
    save_validation_results(auto_vs_expert_results, "auto_box_results.json")
```

### C. Utility Functions For Testing

```python
def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice coefficient between predicted and ground truth masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0

def calculate_box_overlap(boxes1, boxes2):
    """Calculate IoU-based overlap between two sets of bounding boxes."""
    # Match boxes between the two sets
    overlaps = []
    for box1 in boxes1:
        max_overlap = 0
        for box2 in boxes2:
            # Calculate IoU between boxes
            overlap = box_iou(box1, box2)
            max_overlap = max(max_overlap, overlap)
        overlaps.append(max_overlap)
    return np.mean(overlaps) if overlaps else 0

def box_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0
```

## 7. Timeline and Prioritization

1. **Phase 1: Basic Framework Setup** (1-2 days)
   - Set up directory structure
   - Install dependencies
   - Create basic test fixtures

2. **Phase 2: Unit Tests** (2-3 days)
   - Implement component-level tests
   - Create mocks for interdependent components

3. **Phase 3: Integration Tests** (2-3 days)
   - Build tests for component integration
   - Test error handling and edge cases

4. **Phase 4: End-to-End Validation** (3-4 days)
   - Implement comparison with original DrSAM
   - Create visualization tools for debugging
   - Develop performance benchmarks

5. **Phase 5: Automation and CI** (1-2 days)
   - Set up GitHub Actions
   - Configure test reporting
   - Document testing procedures 