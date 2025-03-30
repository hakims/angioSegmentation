# Dr-SAM Implementation Analysis Summary

## Project Overview

### Goal
To build a fully automated segmentation and anomaly detection pipeline for lower extremity angiograms, using and extending the methodology of the Dr-SAM framework. The end goal is to:

- Generate vessel masks, skeletons, and anomaly points (e.g., stenoses, aneurysms)
- Pre-label angiogram frames to assist with annotation (e.g., in CVAT)
- Enable quality benchmarking and clinical insight generation using vascular imaging data

### Original Dr-SAM Methodology
When ensuring consistency with the "original Dr-SAM" methodology, we reference:
- Original research paper (in lit review folder)
- Original Dr-SAM codebase (in dr_sam folder)

#### Key Findings from Original Paper
1. **Problem Statement**:
   - Out-of-the-box SAM produces inaccurate vessel masks
   - Manual bounding box labeling is time-consuming

2. **Dr-SAM Solution**:
   - Uses 3 expert-labeled bounding boxes
   - Implements smart point selection:
     - First point: Blackest pixel in box (likely in contrast-enhanced vessel)
     - Second point: Selected within predefined radius, focusing on vessel continuity
   - Performs 5 iterations of point selection and segmentation
   - Quantifies results using IoU

3. **Our Extension**:
   - Addresses the "no time for pre-labeling" scenario
   - Focuses on lower extremity angiograms
   - Uses Gaussian/Frangi transformation for automated bounding box generation
   - Compares results with expert-labeled boxes

### Project Structure
```
AngioMLM/
├── bin/
│   ├── buildDrSAM.py               # Main runner script: crops, extracts, boxes, segments
│   ├── cleanup.py                  # Main cleanup utility for generated files
│   └── buildDependencies.py       # Handles torch and other setup dependencies
│
├── tests/
│   ├── conftest.py                 # Test configuration and shared fixtures
│   ├── test_single_frame_pipeline.py # Single frame pipeline tests
│   ├── test_schema_validation.py   # Schema validation tests
│   ├── cleanupTests.py            # Test-specific cleanup utility
│   ├── test_utils/
│   │   ├── visualization.py       # Test visualization utilities
│   │   ├── validation_utils.py    # Test validation utilities
│   │   └── file_utils.py          # Test file handling utilities
│   ├── outputs/                   # Test outputs directory
│   │   └── visualizations/        # Test visualization outputs
│   └── test_data/                 # Test data directory
│       ├── single_frame/          # Single frame test cases
│       ├── multiple_frame/        # Multiple frame test cases
│       └── schema/                # Schema validation test cases
│
├── utils/
│   ├── io.py                      # Frame extraction, video cropping, folder scans
│   ├── schema_utils.py            # Schema validation utilities
│   ├── coco_utils.py              # COCO format conversion utilities
│   └── cleanup_utils.py           # Shared cleanup utilities
│
├── segmentation/
│   ├── segment_vessels.py         # Core segmentation wrapper using Dr-SAM
│   └── segmentationPipeline.py    # Saves masks, skeletons, debug images, metadata
│
├── preprocessing/
│   ├── bounding_boxes.py          # Smart bounding box generation using filters + contours
│   └── transforms.py              # Defines transform options: frangi, clahe, hessian
│
└── dr_sam_core/                   # Refactored core from the original Dr-SAM repo
    ├── segmentation/
    │   ├── predictors.py          # SAM model loader and segmentize() logic
    │   └── ...                    # Other supporting segmentation logic
    ├── anomality_detection/
    │   ├── skeletonize.py         # Skeleton extraction logic
    │   ├── detection_algorithms.py# Stenosis / aneurysm detection methods
    │   └── utils.py               # Utility functions for mask/skeleton analysis
    └── checkpoints/
        └── sam_vit_h.pth          # Model weights for Segment Anything
```

## Implementation Status

### Current Features

1. **Testing Framework**
   - Comprehensive pytest suite
   - Organized test cases by type
   - Shared fixtures for common operations
   - Test output handling with visualization support
   - Cleanup utilities

2. **Data Schema**
   - Custom metadata storage schema
   - Schema validation utilities
   - COCO format conversion support
   - Vessel attributes and annotations support

3. **Core Functionality**
   - Frame extraction from videos
   - Image preprocessing with transforms
   - Bounding box generation
   - Vessel segmentation using Dr-SAM
   - Skeleton and anomaly detection
   - Visualization and debug output

4. **Cleanup System**
   - Main cleanup utility
   - Test-specific cleanup utility
   - Visualization preservation support
   - Nested directory handling

### Output Directory Structure
```
├── frames/                        # Extracted image frames (auto-generated)
├── masks/                         # Output vessel masks (black vessel on white bg)
├── skeletons_raw/                # Unpruned skeletons for each mask
├── skeletons/                    # Pruned skeletons after filtering
├── metadata/                     # Anomaly JSON outputs
├── debug/
│   ├── bounding_boxes/           # Bounding box overlays from preprocessing
│   └── segment_vessels/          # Original image overlays with prompt points
```

## Current Issues and TODOs

### 1. Single Frame Pipeline Testing
- [ ] Debug image label production from metadata
- [ ] Verify mask overlay functionality
- [ ] Ensure proper bounding box handling

### 2. Original Dr-SAM Dataset Integration
- [ ] Build and run test suite for original dataset
- [ ] Verify metadata conversion accuracy
- [ ] Ensure Dr-SAM helper function compatibility
- [ ] Test stenosis calculation functionality

### 3. Schema and Labeling
- [ ] Verify accurate label/mask overlay
- [ ] Handle missing bounding box cases
- [ ] Improve mask-only annotation support

### 4. Auto-box Generation
- [ ] Implement automatic bounding box generation
- [ ] Compare with expert-labeled boxes
- [ ] Validate segmentation quality

### 5. Additional Improvements Needed
- [ ] Enhance error handling in test fixtures
- [ ] Add comprehensive logging
- [ ] Improve visualization utilities
- [ ] Better test case organization
- [ ] Add metadata validation checks
- [ ] Implement performance benchmarks
- [ ] Improve test case documentation

## Testing Framework

### Test Structure
```
tests/
├── conftest.py
├── test_single_frame.py
│   └── TestSingleFramePipeline
│       ├── test_metadata_validation
│       ├── test_annotation_creation
│       └── test_stenosis_calculation
├── test_schema_validation.py
│   └── TestSchemaValidation
│       ├── test_basic_validation
└── test_data/
    ├── single_frame/
    │   ├── exampleCase1/
    │   │   ├── image.png
    │   │   └── metadata.json
    │   └── exampleCase2/
    │       ├── image.png
    │       └── metadata.json
    ├── multiple_frame/
    │   └── exampleCase/
    │       ├── images/
    │       └── metadata.json
    └── mp4_angios/
        └── exampleCase/
            ├── media/
            └── metadata.json
```

### Test Case Example
```
single_frame Test Case: "SFA mask and stenosis"
└── TestSingleFramePipeline instance
    ├── test_metadata_validation
    ├── test_annotation_creation
    ├── test_stenosis_calculation
    └── test_additional_features (tbd)
```

### Fixture-based Parametrization
```python
@pytest.fixture(params=get_test_cases())
def test_case(request):
    """Fixture that provides test case data"""
    case_path = Path(__file__).parent / "test_data" / "single_frame" / request.param
    return {
        "name": request.param,
        "path": case_path,
        "metadata": load_metadata(case_path / "metadata.json"),
        "image": load_image(case_path / "image.png")
    }
```

### Benefits of Current Approach
1. **Automatic Discovery**
   - New test cases automatically included
   - No code changes needed for new cases

2. **Clear Organization**
   - Test logic separate from test data
   - Each test case self-contained
   - Easy to add new test cases

3. **Flexible Execution**
   - Run all cases
   - Run specific cases
   - Run specific tests across all cases

4. **Maintainable**
   - Test logic written once
   - Data organized by case
   - Easy to update test cases

## Results Storage
All test results are stored in the `tests/outputs/` folder 