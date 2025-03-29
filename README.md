# Dr-SAM: AI-Powered Angiogram Segmentation

A vessel segmentation and anomaly detection pipeline for lower extremity angiograms, based on the Dr-SAM methodology.

## Overview

This project implements and extends the Dr-SAM (Detecting Retinal vessel with Segment Anything Model) framework for segmenting vessels in angiographic images. Our contributions include:

1. **Automated Vessel Detection**: Added vessel-enhancing filters and automated bounding box generation
2. **Full Pipeline**: Frame extraction, preprocessing, segmentation, skeletonization, and anomaly detection
3. **Testing Framework**: Comprehensive validation utilities for comparing against expert annotations
4. **COCO Format Support**: Added standardized COCO format for annotations and interoperability

## Project Structure

```
├── bin/
│   ├── buildDrSAM.py               # Main runner script
│   ├── cleanupDrSAM.py             # Utility for cleanup
│   ├── buildDependencies.py        # Sets up dependencies
│   └── validateCOCO.py             # COCO format validation utility
│
├── segmentation/
│   ├── segment_vessels.py          # Core Dr-SAM segmentation wrapper
│   └── segmentationPipeline.py     # Batch processing of frames
│
├── preprocessing/
│   ├── bounding_boxes.py           # Automated box generation
│   └── transforms.py               # Vessel-enhancing filters (frangi, etc.)
│
├── dr_sam_core/                    # Core Dr-SAM code
│   ├── segmentation/
│   │   ├── predictors.py           # SAM model loader and segmentation
│   │   └── ...
│   ├── anomality_detection/
│   │   ├── skeletonize.py          # Skeleton extraction
│   │   ├── detection_algorithms.py # Stenosis/aneurysm detection
│   │   └── utils.py
│   └── checkpoints/
│       └── sam_vit_h.pth           # Model weights (auto-downloaded)
│
├── utils/
│   ├── io.py                       # File/video handling utilities
│   ├── coco_utils.py               # COCO format utilities
│   ├── validator.py                # Base validation framework
│   ├── dr_sam_validator.py         # Dr-SAM specific validation 
│   └── dr_sam_runner.py            # Pipeline execution utilities
│
├── tests/
│   ├── originalDrSAM/              # Original Dr-SAM testing utilities
│   └── test_original_drsam.py      # Test script for Dr-SAM validation
│
└── run_tests.py                    # Test runner script
```

## Recent Improvements

1. **Refactored Box Generation**:
   - Unified image processing and bounding box generation
   - Reduced code duplication and improved clarity
   - Added support for both pre-defined and auto-generated boxes

2. **Enhanced Testing Pipeline**:
   - Properly prioritizes expert boxes for validation
   - Improves separation between running and evaluation
   - Added comprehensive validation framework

3. **Improved Code Organization**:
   - Better file structure and module organization
   - Reduced redundancy and circular dependencies
   - Consistent parameter handling

4. **COCO Format Integration**:
   - Added COCO format support for standardized annotations
   - Implemented conversion between Dr-SAM and COCO formats
   - Created validation utilities for COCO annotations
   - Enabled interoperability with CVAT and other annotation tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/drsam-angiogram.git
cd drsam-angiogram

# Install dependencies
pip install -r requirements.txt

# Download SAM weights (will be auto-downloaded on first run)
python bin/buildDependencies.py
```

## Usage

### Processing Angiogram Videos

```bash
# Basic usage (process video folder)
python bin/buildDrSAM.py path/to/angiogram/folder

# With specific transformation method
python bin/buildDrSAM.py path/to/angiogram/folder --method median

# Disable auto box generation (requires custom boxes)
python bin/buildDrSAM.py path/to/angiogram/folder --no-auto-boundingBox --user-provided-boundingBoxes path/to/boxes.json

# Use custom bounding boxes in COCO or Dr-SAM format
python bin/buildDrSAM.py path/to/angiogram/folder --user-provided-boundingBoxes path/to/annotations.json

# Specify output annotation format (drsam, coco, or both)
python bin/buildDrSAM.py path/to/angiogram/folder --output-format coco
```

### COCO Format Utilities

The pipeline now supports COCO format annotations, which are compatible with CVAT and other annotation tools.

```bash
# Convert Dr-SAM format to COCO format
python bin/buildDrSAM.py --convert-to-coco path/to/boundingBoxes.json --image-dir path/to/images --output-json annotations_coco.json

# Convert COCO format to Dr-SAM format
python bin/buildDrSAM.py --convert-to-drsam path/to/annotations_coco.json --output-json boundingBoxes.json

# Validate and analyze COCO format annotations
python bin/validateCOCO.py path/to/annotations_coco.json --validate --image-dir path/to/images

# Visualize COCO annotations
python bin/validateCOCO.py path/to/annotations_coco.json --visualize --image-dir path/to/images --max-images 10

# Compare two COCO annotation files
python bin/validateCOCO.py path/to/annotations1.json --compare path/to/annotations2.json
```

### Bounding Box Priority

The pipeline prioritizes bounding boxes in the following order:

1. **Custom boxes from command line** (`--user-provided-boundingBoxes path/to/boxes.json`)
2. **Local boxes.json** in the processing folder
3. **Auto-generated boxes** using vessel-enhancing filters (if enabled)

If `--no-auto-boundingBox` is specified and no boxes are available, the pipeline will exit with an error.

### Testing Against Original Dr-SAM Dataset

```bash
# Run validation against original Dr-SAM
python run_tests.py original --visualize

# Specify sample size
python run_tests.py original --sample 10

# Only evaluate existing outputs (skip segmentation)
python run_tests.py original --evaluate-only

# Run all tests
python run_tests.py all
```

## Output Structure

Each processed folder will have the following structure:

```
├── frames/                        # Extracted frames
├── masks/                         # Segmented vessel masks
├── skeletons_raw/                 # Unpruned skeletons
├── skeletons/                     # Pruned skeletons 
├── metadata/                      # Anomaly detection JSONs
├── boundingBoxes.json             # Generated bounding boxes (Dr-SAM format)
├── annotations_coco.json          # Generated annotations (COCO format)
├── debug/
│   ├── bounding_boxes/            # Bounding box visualizations
│   └── segment_vessels/           # Segmentation visualizations
```

## Transformation Methods

- `median`: Mode filter (default, original Dr-SAM approach)
- `frangi`: Frangi vessel enhancement filter (recommended for auto box generation)
- `hessian`: Hessian-based vessel enhancement
- `clahe`: Contrast Limited Adaptive Histogram Equalization
- `tophat`: Top-hat morphological operation
- `none`: No transformation

## Validation Metrics

The validation framework calculates:
- IoU (Intersection over Union)
- Dice coefficient

Results are saved in the `tests/results` directory.

## Annotation Formats

The pipeline supports two annotation formats:

### Dr-SAM Format

A simple JSON format mapping filenames to bounding boxes:

```json
{
  "frame_001.png": [[10, 20, 100, 200], [150, 50, 250, 180]],
  "frame_002.png": [[15, 25, 105, 205]]
}
```

### COCO Format

A standardized format compatible with CVAT and other tools:

```json
{
  "info": { "description": "Dr-SAM Angiogram Dataset", "version": "1.0" },
  "categories": [{ "id": 1, "name": "vessel", "supercategory": "vessel" }],
  "images": [
    { "id": 1, "file_name": "frame_001.png", "width": 640, "height": 480 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [10, 20, 90, 180],
      "area": 16200,
      "iscrowd": 0
    }
  ]
}
```

Note: In COCO format, bounding boxes are represented as [x, y, width, height], while Dr-SAM format uses [x1, y1, x2, y2].

## License

[Add appropriate license information]

## References

- Original Dr-SAM paper: [Reference]
- Dr-SAM repository: [Reference]
- Segment Anything Model (SAM): [Reference]
- COCO dataset format: [https://cocodataset.org/#format-data] 