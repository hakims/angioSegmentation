"""
Base Test Class
--------------
This module provides a base test class that all Dr-SAM test modules can inherit from.
It includes common functionality for handling test data, metadata, and validation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pytest

from utils.schema_utils import validate_attributes
from utils.coco_utils import create_coco_skeleton, drsam_to_coco, coco_to_drsam

# Using pytest fixtures instead of class initialization
@pytest.fixture
def test_root():
    """Fixture that provides the root directory for tests."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def visualization_dir():
    """Fixture that provides the visualization directory."""
    # Get the output directory from pytest config
    output_dir = Path(__file__).parent / "outputs"
    vis_dir = output_dir / "visualizations"
    
    # Create directories if they don't exist
    output_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    return vis_dir

class BaseTest:
    """Base class for all Dr-SAM tests."""
    
    def load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        Load metadata from a JSON file.
        
        Args:
            metadata_path: Path to the metadata file
            
        Returns:
            Loaded metadata dictionary
            
        Raises:
            FileNotFoundError: If the metadata file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path) as f:
            return json.load(f)
            
    def save_metadata(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: Metadata dictionary to save
            output_path: Path to save the metadata file
            
        Raises:
            IOError: If there's an error writing the file
        """
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Validate metadata against schema.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = {'name', 'type', 'attributes'}
        missing_fields = required_fields - set(metadata.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            
        # Validate attributes if present
        if 'attributes' in metadata:
            attr_errors = validate_attributes(metadata['attributes'])
            errors.extend(attr_errors)
            
        return errors
        
    def create_test_metadata(self, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a test metadata dictionary.
        
        Args:
            attributes: Optional dictionary of attributes to include
            
        Returns:
            Test metadata dictionary
        """
        metadata = {
            'name': 'test_case',
            'type': 'single_frame',
            'attributes': attributes or {}
        }
        return metadata
        
    def convert_coco_to_drsam(self, coco_data: Dict[str, Any]) -> tuple:
        """
        Convert COCO format data to Dr-SAM format.
        
        Args:
            coco_data: COCO format data dictionary
            
        Returns:
            Tuple of (boxes, attributes)
        """
        return coco_to_drsam(coco_data)
        
    def convert_drsam_to_coco(self, 
                            drsam_boxes: Dict[str, List[List[int]]],
                            image_dir: Path,
                            attributes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Convert Dr-SAM format data to COCO format.
        
        Args:
            drsam_boxes: Dr-SAM format boxes dictionary
            image_dir: Directory containing images
            attributes: Optional dictionary of attributes
            
        Returns:
            COCO format data dictionary
        """
        return drsam_to_coco(drsam_boxes, image_dir, attributes) 