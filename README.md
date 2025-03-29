# Dr-SAM: AI-Powered Angiogram Segmentation

A vessel segmentation and anomaly detection pipeline for lower extremity angiograms, based on the Dr-SAM methodology.

## Overview

This project implements and extends the Dr-SAM (Detecting Retinal vessel with Segment Anything Model) framework for segmenting vessels in angiographic images. Our contributions include:

1. **Automated Vessel Detection**: Added vessel-enhancing filters and automated bounding box generation
2. **Full Pipeline**: Frame extraction, preprocessing, segmentation, skeletonization, and anomaly detection
3. **Testing Framework**: Comprehensive validation utilities for comparing against expert annotations

## Project Structure

```
├── bin/
│   ├── buildDrSAM.py               # Main runner script
│   ├── cleanupDrSAM.py             # Utility for cleanup
│   └── buildDependencies.py        # Sets up dependencies 
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
│   ├── validator.py                # Base validation framework
│   ├── dr_sam_validator.py         # Dr-SAM specific validation 
│   └── dr_sam_runner.py            # Pipeline execution utilities
│
├── tests/
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
python bin/buildDrSAM.py path/to/angiogram/folder --no-auto-box --custom-boxes path/to/boxes.json

# Use custom bounding boxes
python bin/buildDrSAM.py path/to/angiogram/folder --custom-boxes path/to/boxes.json
```

### Bounding Box Priority

The pipeline prioritizes bounding boxes in the following order:

1. **Custom boxes from command line** (`--custom-boxes path/to/boxes.json`)
2. **Local boxes.json** in the processing folder
3. **Auto-generated boxes** using vessel-enhancing filters (if enabled)

If `--no-auto-box` is specified and no boxes are available, the pipeline will exit with an error.

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
├── debug/
│   ├── bounding_boxes/            # Bounding box visualizations
│   └── segment_vessels/           # Segmentation visualizations
└── boxes.json                     # Generated bounding boxes
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

## License

[Add appropriate license information]

## References

- Original Dr-SAM paper: [Reference]
- Dr-SAM repository: [Reference]
- Segment Anything Model (SAM): [Reference] 