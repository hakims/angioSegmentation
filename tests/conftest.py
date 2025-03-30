"""
Test Configuration and Fixtures
-----------------------------
This module provides common fixtures and utilities for all Dr-SAM tests.
It leverages existing utilities from the utils directory to avoid code duplication.

Important Notes:
---------------
1. Test Output Cleanup:
   - By default, test outputs and visualizations are automatically cleaned up after tests complete
   - Use --keep-visualizations flag to preserve outputs for inspection
   - Example: pytest tests/test_single_frame_pipeline.py --keep-visualizations
   - Outputs will be saved in tests/outputs/visualizations/

2. Output Directory Structure:
   - Each test gets a unique output directory based on its ID
   - Visualizations are stored in tests/outputs/visualizations/<test_id>/
   - Other test outputs are stored in tests/outputs/<test_id>/

3. Cleanup Behavior:
   - Without --keep-visualizations: All outputs are deleted after tests
   - With --keep-visualizations: Outputs are preserved for manual inspection
   - Use cleanupTests.py to manually clean up preserved outputs
"""

import os
import sys
import pytest
import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.schema_utils import get_schema
from utils.coco_utils import create_coco_skeleton, drsam_to_coco, coco_to_drsam
from tests.test_utils.validation_utils import validate_metadata, load_and_validate_metadata
from tests.test_utils.file_utils import find_image_files

def pytest_addoption(parser):
    """
    Add custom command line options to pytest.
    
    Options:
    --------
    --keep-visualizations
        Keep test outputs and visualizations after tests complete.
        This is useful when you need to inspect the outputs manually.
        
        Example:
            pytest tests/test_single_frame_pipeline.py --keep-visualizations
            
        Outputs will be saved in:
            - tests/outputs/visualizations/<test_id>/ for visualizations
            - tests/outputs/<test_id>/ for other test outputs
            
        Note: Use cleanupTests.py to manually clean up preserved outputs.
    """
    parser.addoption(
        "--keep-visualizations",
        action="store_true",
        default=False,
        help="Keep visualization files after tests complete (don't clean up)"
    )

def pytest_configure(config):
    """
    Register custom configuration options with pytest.
    This prevents warnings about unknown config options.
    """
    # Register our custom markers
    config.addinivalue_line(
        "markers",
        "single_frame: marks tests as single frame pipeline tests"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "end_to_end: marks tests as end-to-end tests"
    )

# Define paths as regular variables
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "outputs"
VISUALIZATION_DIR = TEST_OUTPUT_DIR / "visualizations"

@pytest.fixture
def test_root():
    """Fixture that provides the root directory for tests."""
    return TEST_DATA_DIR

@pytest.fixture
def output_dir(request):
    """
    Fixture to provide an output directory for test artifacts.
    
    Creates a unique directory for each test based on the test ID.
    Directory is cleaned up after test unless --keep-visualizations is specified.
    
    Args:
        request: pytest request object containing test information
        
    Returns:
        Path: Directory path for test outputs
        
    Note:
        - Outputs are automatically cleaned up unless --keep-visualizations is used
        - Use --keep-visualizations when you need to inspect the outputs
        - Outputs are stored in tests/outputs/<test_id>/
    """
    # Get unique test ID (replacing characters that aren't valid in paths)
    test_id = request.node.nodeid.replace("/", "_").replace("::", "_").replace("[", "_").replace("]", "_")
    
    # Create output directory
    output_dir = TEST_OUTPUT_DIR / test_id
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Return the directory
    yield output_dir
    
    # Clean up after test unless --keep-visualizations is set
    if not request.config.getoption("--keep-visualizations", False):
        shutil.rmtree(output_dir, ignore_errors=True)

@pytest.fixture
def visualization_dir(request):
    """
    Fixture to provide a visualization directory.
    
    Creates a unique directory for each test based on the test ID.
    Directory is cleaned up after test unless --keep-visualizations is specified.
    
    Args:
        request: pytest request object containing test information
        
    Returns:
        Path: Directory path for test visualizations
        
    Note:
        - Visualizations are automatically cleaned up unless --keep-visualizations is used
        - Use --keep-visualizations when you need to inspect the visualizations
        - Visualizations are stored in tests/outputs/visualizations/<test_id>/
    """
    # Get unique test ID (replacing characters that aren't valid in paths)
    test_id = request.node.nodeid.replace("/", "_").replace("::", "_").replace("[", "_").replace("]", "_")
    
    # Create visualization directory
    vis_dir = VISUALIZATION_DIR / test_id
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # Return the directory
    yield vis_dir
    
    # Clean up after test unless --keep-visualizations is set
    if not request.config.getoption("--keep-visualizations", False):
        shutil.rmtree(vis_dir, ignore_errors=True)

@pytest.fixture
def schema():
    """Fixture to provide the schema for validation."""
    return get_schema()

@pytest.fixture(params=os.listdir(TEST_DATA_DIR / "single_frame") if (TEST_DATA_DIR / "single_frame").exists() else [])
def single_frame_case(request, schema):
    """
    Fixture that provides single frame test cases.
    
    This fixture is parameterized based on the contents of test_data/single_frame.
    Each subdirectory is treated as a test case.
    """
    case_path = TEST_DATA_DIR / "single_frame" / request.param
    if not case_path.is_dir():
        pytest.skip(f"Test case '{request.param}' is not a directory")
    
    # Load metadata if exists
    metadata_path = case_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            pytest.skip(f"Invalid JSON in metadata file: {metadata_path}")
    
    # Find any image file in the directory using our utility function
    image_files = find_image_files(case_path)
    
    # Load the first image if any were found
    image = None
    if image_files:
        try:
            image = cv2.imread(str(image_files[0]))
        except Exception as e:
            pytest.skip(f"Failed to read image file: {image_files[0]}, error: {e}")
    
    return {
        "name": request.param,
        "path": case_path,
        "metadata": metadata,
        "schema": schema,
        "image": image,
        "image_path": image_files[0] if image_files else None
    }

@pytest.fixture(params=os.listdir(TEST_DATA_DIR / "multiple_frame") if (TEST_DATA_DIR / "multiple_frame").exists() else [])
def multiple_frame_case(request, schema):
    """
    Fixture that provides multiple frame test cases.
    
    This fixture is parameterized based on the contents of test_data/multiple_frame.
    Each subdirectory is treated as a test case.
    """
    case_path = TEST_DATA_DIR / "multiple_frame" / request.param
    if not case_path.is_dir():
        pytest.skip(f"Test case '{request.param}' is not a directory")
    
    # Load metadata if exists
    metadata_path = case_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            pytest.skip(f"Invalid JSON in metadata file: {metadata_path}")
    
    # Load images from either:
    # 1. An 'images' subdirectory
    # 2. Any image files directly in the case directory
    images = []
    image_paths = []
    
    # Look in 'images' subdirectory first
    images_dir = case_path / "images"
    if images_dir.exists() and images_dir.is_dir():
        image_files = find_image_files(images_dir)
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    image_paths.append(img_path)
            except Exception as e:
                print(f"Warning: Failed to read image file: {img_path}, error: {e}")
    
    # If no images found in subdirectory, look for images directly in the case directory
    if not images:
        image_files = find_image_files(case_path)
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append(img)
                    image_paths.append(img_path)
            except Exception as e:
                print(f"Warning: Failed to read image file: {img_path}, error: {e}")
    
    return {
        "name": request.param,
        "path": case_path,
        "metadata": metadata,
        "schema": schema,
        "images": images,
        "image_paths": image_paths
    }

@pytest.fixture(params=os.listdir(TEST_DATA_DIR / "schema") if (TEST_DATA_DIR / "schema").exists() else [])
def schema_test_case(request, schema):
    """
    Fixture that provides schema validation test cases.
    
    This fixture is parameterized based on the contents of test_data/schema.
    Each subdirectory is treated as a test case.
    """
    case_path = TEST_DATA_DIR / "schema" / request.param
    if not case_path.is_dir():
        pytest.skip(f"Test case '{request.param}' is not a directory")
    
    # Load metadata if exists
    metadata_path = case_path / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            pytest.skip(f"Invalid JSON in metadata file: {metadata_path}")
    
    # Load expected validation result if exists
    expected_path = case_path / "expected.json"
    expected = {}
    if expected_path.exists():
        try:
            with open(expected_path) as f:
                expected = json.load(f)
        except json.JSONDecodeError:
            pytest.skip(f"Invalid JSON in expected file: {expected_path}")
    
    return {
        "name": request.param,
        "path": case_path,
        "metadata": metadata,
        "expected": expected,
        "schema": schema
    }

#
# Test case discovery functions
#

def get_schema_test_cases() -> List[str]:
    """
    Get list of schema validation test cases.
    
    Returns:
        List of test case directory names
    """
    cases_dir = project_root / "test_data" / "schema_validation"
    if not cases_dir.exists():
        return []
    return [d.name for d in cases_dir.iterdir() if d.is_dir()]

def get_single_frame_cases() -> List[str]:
    """
    Get list of single frame test cases.
    
    Returns:
        List of test case directory names
    """
    cases_dir = project_root / "test_data" / "single_frame"
    if not cases_dir.exists():
        return []
    return [d.name for d in cases_dir.iterdir() if d.is_dir()]

def get_multiple_frame_cases() -> List[str]:
    """
    Get list of multiple frame test cases.
    
    Returns:
        List of test case directory names
    """
    cases_dir = project_root / "test_data" / "multiple_frame"
    if not cases_dir.exists():
        return []
    return [d.name for d in cases_dir.iterdir() if d.is_dir()]

#
# Test data loading functions
#

def load_metadata(path: Path) -> Dict[str, Any]:
    """
    Load metadata from JSON file.
    
    Args:
        path: Path to the metadata file
        
    Returns:
        Dictionary containing the metadata (empty if file not found)
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        path: Path to the image file
        
    Returns:
        NumPy array containing the image (None if file not found)
    """
    if not path.exists():
        return None
    return cv2.imread(str(path))

def load_images(dir_path: Path) -> List[np.ndarray]:
    """
    Load all images from directory.
    
    Args:
        dir_path: Path to the directory containing images
        
    Returns:
        List of NumPy arrays containing the images (empty if directory not found)
    """
    if not dir_path.exists():
        return []
    return [load_image(p) for p in dir_path.glob("*.png")]

#
# Legacy fixtures for backward compatibility
#

@pytest.fixture
def get_test_case(test_data_root) -> callable:
    """
    Get a test case directory by name.
    
    Args:
        test_case: Name of the test case directory
        
    Returns:
        Path to the test case directory
        
    Raises:
        FileNotFoundError: If the test case directory doesn't exist
    """
    def _get_test_case(test_case: str) -> Path:
        # Try single frame first
        test_dir = test_data_root / "single_frame" / test_case
        if test_dir.exists():
            return test_dir
            
        # Try multiple frame
        test_dir = test_data_root / "multiple_frame" / test_case
        if test_dir.exists():
            return test_dir
            
        # Try schema validation
        test_dir = test_data_root / "schema_validation" / test_case
        if test_dir.exists():
            return test_dir
            
        raise FileNotFoundError(f"Test case directory not found: {test_case}")
    return _get_test_case

@pytest.fixture
def create_test_metadata(schema) -> callable:
    """
    Create a test metadata file with valid attributes.
    
    Args:
        test_dir: Directory to create the metadata file in
        attributes: Optional dictionary of attributes to include
        
    Returns:
        Path to the created metadata file
    """
    def _create_test_metadata(test_dir: Path, attributes: Optional[Dict[str, Any]] = None) -> Path:
        # Create COCO format metadata
        coco_data = create_coco_skeleton()
        
        # Add test image
        coco_data["images"].append({
            "id": 1,
            "file_name": "test_image.png",
            "width": 800,
            "height": 600,
            "date_captured": "2024-03-29 00:00:00",
            "license": 1,
        })
        
        # Add test annotation with valid attributes
        test_attributes = {
            "vessel_id": "SFA",
            "side": "R",
            "segment": "proximal",
            "annotator": "test",
            "label_schema_version": "2025.03"
        }
        if attributes:
            test_attributes.update(attributes)
            
        coco_data["annotations"].append({
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 200, 200],
            "area": 40000,
            "segmentation": [],
            "iscrowd": 0,
            "attributes": test_attributes
        })
        
        # Save metadata file
        metadata_path = test_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        return metadata_path
    return _create_test_metadata

@pytest.fixture
def validate_test_metadata(schema) -> callable:
    """
    Validate test metadata against the schema.
    
    Args:
        metadata: Metadata to validate
        custom_schema: Optional schema to validate against (uses default if not provided)
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    def _validate_test_metadata(
        metadata: Union[Dict[str, Any], Path], 
        custom_schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        schema_to_use = custom_schema if custom_schema else schema
        
        if isinstance(metadata, Path):
            # Load metadata from file
            metadata_dict, is_valid, errors = load_and_validate_metadata(metadata, schema_to_use)
            return is_valid, errors
        else:
            # Use provided metadata dictionary
            is_valid, errors = validate_metadata(metadata, schema_to_use)
            return is_valid, errors
            
    return _validate_test_metadata 