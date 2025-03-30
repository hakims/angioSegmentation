# Dr-SAM Testing Framework

This directory contains the testing framework for the Dr-SAM angiogram segmentation pipeline. The framework is built on pytest and provides a flexible, maintainable way to test different aspects of the pipeline.

## Test Organization

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── base_test.py            # Base test class with common functionality
├── test_single_frame_pipeline.py  # Single frame processing tests
├── test_multiple_frame_pipeline.py # Multiple frame processing tests
├── test_schema_validation.py # Schema validation tests
├── test_drsam_validation.py  # Tests against original Dr-SAM
├── test_original_drsam.py    # Original Dr-SAM integration tests
├── cleanupTests.py          # Test cleanup utility
├── test_utils/              # Shared testing utilities
│   ├── visualization.py     # Visualization utilities
│   ├── validation_utils.py  # Schema and metadata validation
│   └── file_utils.py        # File handling utilities
├── outputs/                 # Test outputs directory
│   └── visualizations/      # Test visualization outputs
└── test_data/              # Test data directory
    ├── single_frame/       # Single frame test cases
    ├── multiple_frame/     # Multiple frame test cases
    └── schema/             # Schema validation test cases
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_single_frame_pipeline.py

# Run specific test function
pytest tests/test_single_frame_pipeline.py::TestSingleFramePipeline::test_metadata_validation
```

### Test Categories and Markers

```bash
# Run single frame pipeline tests
pytest -m single_frame

# Run multiple frame pipeline tests
pytest -m multiple_frame

# Run schema validation tests
pytest -m schema

# Run original DrSAM tests
pytest -m original_drsam
```

### Useful Command Line Options

```bash
# Keep visualization outputs after tests
pytest --keep-visualizations

# Show test coverage
pytest --cov=.

# Run tests in parallel
pytest -n auto

# Show warnings
pytest -W always
```

## Test Output Management

### Visualization Outputs

By default, test outputs and visualizations are automatically cleaned up after tests complete. To preserve outputs for inspection:

```bash
# Keep visualization outputs
pytest --keep-visualizations

# Outputs will be saved in tests/outputs/visualizations/<test_id>/
```

### Manual Cleanup

Use the cleanup utility to manually clean up preserved outputs:

```bash
# Clean all test outputs
python tests/cleanupTests.py

# Clean specific directory
python tests/cleanupTests.py path/to/directory

# Keep visualizations while cleaning
python tests/cleanupTests.py --keep-visualizations
```

## Creating Test Cases

### Test Case Structure

The framework automatically discovers test cases based on directory structure. Each test case should be organized as follows:

#### Single Frame Test Case
```
tests/test_data/single_frame/my_test_case/
├── metadata.json    # COCO format metadata
└── image file       # Any common image format (png, jpg, tif, etc.)
```

#### Multiple Frame Test Case
```
tests/test_data/multiple_frame/my_sequence_test/
├── metadata.json       # COCO format metadata
└── images/             # Directory containing image sequence
    ├── frame001.png
    ├── frame002.png
    └── frame003.png
```

#### Schema Validation Test Case
```
tests/test_data/schema/attribute_validation_test/
├── metadata.json    # Metadata to validate
└── expected.json    # Expected validation results
```

### Test Fixtures

The framework provides several fixtures in `conftest.py`:

1. Core Fixtures:
   - `test_root`: Root directory for test data
   - `output_dir`: Directory for test outputs (unique per test)
   - `visualization_dir`: Directory for test visualizations
   - `schema`: CVAT labeling schema

2. Test Case Fixtures:
   - `single_frame_case`: Provides single frame test cases
   - `multiple_frame_case`: Provides multiple frame test cases
   - `schema_test_case`: Provides schema validation test cases

3. Utility Fixtures:
   - `create_test_metadata`: Creates test metadata files
   - `validate_test_metadata`: Validates metadata against schema

### Writing Tests

1. Basic Test Structure:
```python
import pytest
from tests.base_test import BaseTest

class TestSingleFramePipeline(BaseTest):
    def test_metadata_validation(self, single_frame_case):
        # Test logic here
        pass

    def test_annotation_creation(self, single_frame_case):
        # Test logic here
        pass
```

2. Using Fixtures:
```python
def test_with_visualization(self, single_frame_case, visualization_dir):
    # Test logic here
    # Save visualization to visualization_dir
    pass
```

3. Parameterized Tests:
```python
@pytest.mark.parametrize("transform_type", ["frangi", "clahe", "hessian"])
def test_transforms(self, single_frame_case, transform_type):
    # Test logic here
    pass
```

## Best Practices

1. Test Organization:
   - Group related tests in test classes
   - Use descriptive test names
   - Keep test logic focused and simple

2. Test Data:
   - Use realistic test data
   - Include edge cases
   - Document test data requirements

3. Output Management:
   - Use `--keep-visualizations` for debugging
   - Clean up outputs after debugging
   - Use unique test IDs for outputs

4. Error Handling:
   - Test both success and failure cases
   - Use appropriate assertions
   - Include meaningful error messages

## Common Issues and Solutions

1. Test Output Cleanup:
   - Use `--keep-visualizations` to preserve outputs
   - Run `cleanupTests.py` to clean up manually
   - Check `tests/outputs/` for preserved files

2. Test Discovery:
   - Ensure test files start with `test_`
   - Ensure test functions start with `test_`
   - Check test markers are properly registered

3. Fixture Usage:
   - Use appropriate fixtures for test cases
   - Avoid modifying fixture data
   - Clean up after using fixtures

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [COCO Format Documentation](https://cocodataset.org/#format-data)
- [Dr-SAM Paper](path/to/drsam/paper) 