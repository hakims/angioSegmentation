# Dr-SAM Testing Framework

This directory contains tests and testing utilities for the Dr-SAM angiogram segmentation pipeline.

## Getting Started

The testing framework has a single entry point: `testingMaster.py`.

```bash
# Run the original Dr-SAM validation test (default)
python tests/testingMaster.py 

# Run with specific options
python tests/testingMaster.py drsam --method frangi --visualize
```

## Testing Structure

- `testingMaster.py`: Main entry point for all tests
- `utils/`: Testing utilities for validation and pipeline execution
  - `testing_utils.py`: Shared testing utilities (metrics, visualization, pipeline execution)
  - `dr_sam_validator.py`: Validator specifically for the original Dr-SAM dataset

## Test Types

Currently supported test types:

- `drsam`: Validate against the original Dr-SAM dataset (default)
  ```bash
  python tests/testingMaster.py drsam [options]
  ```

Future test types (to be implemented):
- `performance`: Benchmarking and performance tests
- `integration`: End-to-end integration tests

## Common Options

- `--dataset PATH`: Path to dataset directory (default: dr_sam/dataset)
- `--output-dir PATH`: Output directory for test results (default: tests/outputs)
- `--method METHOD`: Image transformation method to use (default: median)
- `--sample N`: Number of images to sample (default: all)
- `--verbose`: Enable verbose output
- `--visualize`: Generate visualizations of results
- `--skip-pipeline`: Skip pipeline execution and only evaluate existing results
- `--skip-evaluation`: Skip evaluation of results
- `--boundingBoxes-file PATH`: Path to user-provided boundingBoxes JSON file

## Dataset Structure

The `dr_sam/dataset` directory should have the following structure:

```
dr_sam/dataset/
├── images/        # Original DrSAM dataset images
├── masks/         # Reference masks
└── metadata.json  # Expert-labeled bounding boxes
```

## Output Structure

Test outputs are saved to the specified output directory (default: tests/outputs):

```
tests/outputs/
├── debug/             # Debug visualizations
│   └── bounding_boxes/
├── frames/            # Processed frames
├── masks/             # Generated vessel masks
├── metadata/          # Generated metadata
├── skeletons/         # Vessel skeletons
└── skeletons_raw/     # Raw skeletons
```

## Evaluating Results

Results from tests are saved to the `tests/results` directory:

```
tests/results/
└── drsam_validation_results_XX_images.json  # Metrics for each image
```

Visualizations are saved to the `tests/visualizations` directory if the `--visualize` flag is used:

```
tests/visualizations/
└── {image_id}_comparison.png  # Side-by-side comparison images
``` 