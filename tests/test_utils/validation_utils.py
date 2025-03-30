#!/usr/bin/env python
# File: tests/test_utils/validation_utils.py
# Purpose: Common validation utilities for testing

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

def validate_metadata(metadata: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate metadata against a schema.
    
    Args:
        metadata: Metadata dictionary to validate
        schema: Schema dictionary to validate against
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = ["images", "annotations", "categories"]
    for field in required_fields:
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
    
    # If basic structure is invalid, return early
    if errors:
        return False, errors
    
    # Validate images
    for i, img in enumerate(metadata["images"]):
        img_errors = validate_image(img, schema)
        if img_errors:
            errors.extend([f"Image {i} ({img.get('id', 'unknown')}): {err}" for err in img_errors])
            
    # Validate annotations
    for i, ann in enumerate(metadata["annotations"]):
        ann_errors = validate_annotation(ann, schema)
        if ann_errors:
            errors.extend([f"Annotation {i} ({ann.get('id', 'unknown')}): {err}" for err in ann_errors])
    
    # Validate categories
    for i, cat in enumerate(metadata["categories"]):
        cat_errors = validate_category(cat, schema)
        if cat_errors:
            errors.extend([f"Category {i} ({cat.get('id', 'unknown')}): {err}" for err in cat_errors])
    
    return len(errors) == 0, errors

def validate_image(image: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate a single image entry against the schema.
    
    Args:
        image: Image dictionary to validate
        schema: Schema dictionary to validate against
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["id", "file_name", "width", "height"]
    for field in required_fields:
        if field not in image:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    if "id" in image and not isinstance(image["id"], int):
        errors.append(f"Field 'id' must be an integer, got: {type(image['id']).__name__}")
        
    if "width" in image and not isinstance(image["width"], int):
        errors.append(f"Field 'width' must be an integer, got: {type(image['width']).__name__}")
        
    if "height" in image and not isinstance(image["height"], int):
        errors.append(f"Field 'height' must be an integer, got: {type(image['height']).__name__}")
        
    return errors

def validate_annotation(annotation: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate a single annotation entry against the schema.
    
    Args:
        annotation: Annotation dictionary to validate
        schema: Schema dictionary to validate against
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["id", "image_id", "category_id", "bbox"]
    for field in required_fields:
        if field not in annotation:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    if "id" in annotation and not isinstance(annotation["id"], int):
        errors.append(f"Field 'id' must be an integer, got: {type(annotation['id']).__name__}")
        
    if "image_id" in annotation and not isinstance(annotation["image_id"], int):
        errors.append(f"Field 'image_id' must be an integer, got: {type(annotation['image_id']).__name__}")
        
    if "category_id" in annotation and not isinstance(annotation["category_id"], int):
        errors.append(f"Field 'category_id' must be an integer, got: {type(annotation['category_id']).__name__}")
    
    # Validate bbox format
    if "bbox" in annotation:
        bbox = annotation["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"Field 'bbox' must be a list of 4 values, got: {bbox}")
        else:
            for i, val in enumerate(bbox):
                if not isinstance(val, (int, float)):
                    errors.append(f"Bbox value at index {i} must be a number, got: {type(val).__name__}")
    
    # Validate attributes if present
    if "attributes" in annotation:
        attr_errors = validate_attributes(annotation["attributes"], schema)
        errors.extend(attr_errors)
    
    return errors

def validate_category(category: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate a single category entry against the schema.
    
    Args:
        category: Category dictionary to validate
        schema: Schema dictionary to validate against
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["id", "name"]
    for field in required_fields:
        if field not in category:
            errors.append(f"Missing required field: {field}")
    
    # Validate field types
    if "id" in category and not isinstance(category["id"], int):
        errors.append(f"Field 'id' must be an integer, got: {type(category['id']).__name__}")
    
    return errors

def validate_attributes(attributes: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """
    Validate attributes against the schema.
    
    Args:
        attributes: Attributes dictionary to validate
        schema: Schema dictionary to validate against
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Get attribute constraints from schema
    attribute_constraints = schema.get("attribute_constraints", {})
    required_attributes = schema.get("required_attributes", [])
    
    # Check required attributes
    for attr in required_attributes:
        if attr not in attributes:
            errors.append(f"Missing required attribute: {attr}")
    
    # Validate each attribute
    for attr_name, attr_value in attributes.items():
        if attr_name in attribute_constraints:
            constraints = attribute_constraints[attr_name]
            
            # Check allowed values
            if "allowed_values" in constraints:
                allowed_values = constraints["allowed_values"]
                if attr_value not in allowed_values:
                    errors.append(f"Invalid value for attribute '{attr_name}': {attr_value}. "
                                 f"Allowed values: {allowed_values}")
            
            # Check data type
            if "type" in constraints:
                expected_type = constraints["type"]
                if expected_type == "string" and not isinstance(attr_value, str):
                    errors.append(f"Attribute '{attr_name}' must be a string, got: {type(attr_value).__name__}")
                elif expected_type == "number" and not isinstance(attr_value, (int, float)):
                    errors.append(f"Attribute '{attr_name}' must be a number, got: {type(attr_value).__name__}")
                elif expected_type == "boolean" and not isinstance(attr_value, bool):
                    errors.append(f"Attribute '{attr_name}' must be a boolean, got: {type(attr_value).__name__}")
    
    return errors

def load_and_validate_metadata(metadata_path: Union[str, Path], schema: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Load and validate metadata from a file.
    
    Args:
        metadata_path: Path to the metadata file
        schema: Schema dictionary to validate against
        
    Returns:
        Tuple of (metadata, is_valid, list_of_errors)
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        return {}, False, [f"Metadata file not found: {metadata_path}"]
    
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        return {}, False, [f"Failed to parse metadata file: {e}"]
    except Exception as e:
        return {}, False, [f"Error reading metadata file: {e}"]
    
    is_valid, errors = validate_metadata(metadata, schema)
    return metadata, is_valid, errors 