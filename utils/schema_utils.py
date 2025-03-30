"""
Schema Validation Utilities for Dr-SAM
--------------------------------------
This module provides utilities for validating annotation metadata against 
the CVAT labeling schema defined in data_schema/cvat_labeling_schema_2025.03.json.

It includes functions for:
1. Loading the schema
2. Validating attributes against the schema
3. Generating default attributes according to the schema
"""

import json
import os
import jsonschema
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Find the schema file
def get_schema_path() -> Path:
    """Get the path to the CVAT labeling schema file."""
    # Try to find it in the data_schema folder
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    
    schema_paths = [
        repo_root / "data_schema" / "cvat_labeling_schema_2025.03.json",
        # Add fallback locations if needed
    ]
    
    for path in schema_paths:
        if path.exists():
            return path
            
    # If not found, raise an error
    raise FileNotFoundError(f"Could not find schema file in expected locations: {schema_paths}")

def load_schema() -> List[Dict[str, Any]]:
    """Load the CVAT labeling schema."""
    schema_path = get_schema_path()
    with open(schema_path, 'r') as f:
        return json.load(f)

# Cache for the loaded schema to avoid repeated file I/O
_schema_cache = None

def get_schema() -> List[Dict[str, Any]]:
    """Get the CVAT labeling schema, with caching."""
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = load_schema()
    return _schema_cache

def get_category_schema(category: str) -> Optional[Dict[str, Any]]:
    """Get the schema for a specific category."""
    schema = get_schema()
    for item in schema:
        if item["name"] == category:
            return item
    return None

def get_attribute_schema(category: str, attribute: str) -> Optional[Dict[str, Any]]:
    """Get the schema for a specific attribute within a category."""
    category_schema = get_category_schema(category)
    if category_schema and "attributes" in category_schema:
        for attr in category_schema["attributes"]:
            if attr["name"] == attribute:
                return attr
    return None

def get_valid_values(category: str, attribute: str) -> List[str]:
    """Get the valid values for a specific attribute within a category."""
    attr_schema = get_attribute_schema(category, attribute)
    if attr_schema and "values" in attr_schema:
        return attr_schema["values"]
    return []

def get_default_value(category: str, attribute: str) -> str:
    """Get the default value for a specific attribute within a category."""
    attr_schema = get_attribute_schema(category, attribute)
    if attr_schema and "default_value" in attr_schema:
        return attr_schema["default_value"]
    return ""

def get_default_attributes(category: str = "vessel") -> Dict[str, str]:
    """Get default attributes for a category based on the schema."""
    category_schema = get_category_schema(category)
    if not category_schema or "attributes" not in category_schema:
        return {}
        
    defaults = {}
    for attr in category_schema["attributes"]:
        if "name" in attr and "default_value" in attr:
            defaults[attr["name"]] = attr["default_value"]
    
    return defaults

def validate_attribute(category: str, attribute: str, value: str) -> Tuple[bool, str]:
    """
    Validate an attribute value against the schema.
    
    Args:
        category: The category name ('vessel', 'bone', or 'device')
        attribute: The attribute name
        value: The attribute value to validate
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    attr_schema = get_attribute_schema(category, attribute)
    if not attr_schema:
        return False, f"Unknown attribute: {attribute} for category: {category}"
    
    valid_values = attr_schema.get("values", [])
    if not valid_values:
        # If no values are specified, any value is valid
        return True, ""
    
    if value not in valid_values:
        return False, f"Invalid value '{value}' for {category}.{attribute}. Valid values: {', '.join(valid_values)}"
    
    return True, ""

def validate_attributes(category: str, attributes: Dict[str, str]) -> List[str]:
    """
    Validate all attributes for a category against the schema.
    
    Args:
        category: The category name ('vessel', 'bone', or 'device')
        attributes: Dictionary of attribute names to values
    
    Returns:
        List of error messages (empty if all valid)
    """
    errors = []
    
    # First, check that all required attributes are present
    category_schema = get_category_schema(category)
    if not category_schema:
        return [f"Unknown category: {category}"]
    
    # Validate each attribute
    for attr_name, attr_value in attributes.items():
        is_valid, error = validate_attribute(category, attr_name, attr_value)
        if not is_valid:
            errors.append(error)
    
    return errors

def fix_attributes(category: str, attributes: Dict[str, str]) -> Dict[str, str]:
    """
    Fix attributes by replacing invalid values with defaults and adding missing attributes.
    
    Args:
        category: The category name ('vessel', 'bone', or 'device')
        attributes: Dictionary of attribute names to values
    
    Returns:
        Fixed attributes dictionary
    """
    fixed_attributes = attributes.copy()
    
    # Get default attributes for the category
    defaults = get_default_attributes(category)
    
    # Add any missing attributes with default values
    for attr_name, default_value in defaults.items():
        if attr_name not in fixed_attributes:
            fixed_attributes[attr_name] = default_value
    
    # Fix invalid values
    for attr_name, attr_value in list(fixed_attributes.items()):
        is_valid, _ = validate_attribute(category, attr_name, attr_value)
        if not is_valid:
            # Replace with default value
            default_value = get_default_value(category, attr_name)
            fixed_attributes[attr_name] = default_value
    
    return fixed_attributes

def validate_and_fix_attributes_dict(attributes_dict: Dict[str, Dict[int, Dict[str, str]]], 
                                  category: str = "vessel") -> Tuple[Dict[str, Dict[int, Dict[str, str]]], Dict[str, int]]:
    """
    Validate and fix all attributes in an attributes dictionary.
    
    Args:
        attributes_dict: Dictionary mapping filenames to box indices to attributes
        category: Category name to validate against (default: "vessel")
        
    Returns:
        Tuple containing:
        - Updated attributes dictionary with fixed values
        - Dictionary with validation statistics
    """
    if not attributes_dict:
        return attributes_dict, {"total_files": 0, "total_boxes": 0, "total_issues": 0, "fixed_issues": 0}
    
    # Make a copy to avoid modifying the original
    fixed_attributes = {filename: attrs.copy() for filename, attrs in attributes_dict.items()}
    
    # Track validation stats
    stats = {
        "total_files": len(fixed_attributes),
        "total_boxes": sum(len(attrs) for attrs in fixed_attributes.values()),
        "total_issues": 0,
        "fixed_issues": 0
    }
    
    # Validate each file's attributes
    for filename, file_attrs in fixed_attributes.items():
        for box_idx, attrs in list(file_attrs.items()):
            # Validate against specified category
            errors = validate_attributes(category, attrs)
            
            if errors:
                stats["total_issues"] += len(errors)
                
                # Fix attributes
                fixed_attrs = fix_attributes(category, attrs)
                fixed_attributes[filename][box_idx] = fixed_attrs
                stats["fixed_issues"] += 1
    
    return fixed_attributes, stats

def validate_and_fix_coco_annotations(coco_data: Dict[str, Any], source="generic") -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and fix all annotations in a COCO format dictionary.
    
    Args:
        coco_data: COCO format dictionary
        source: Source of the COCO data ('generic', 'cvat', 'originaldrsam')
        
    Returns:
        Tuple containing (fixed_coco_data, list_of_errors)
    """
    errors = []
    fixed_coco = coco_data.copy()
    
    # Ensure categories match our schema
    category_names = [cat.get("name") for cat in fixed_coco.get("categories", [])]
    if "vessel" not in category_names:
        errors.append("Missing required 'vessel' category - adding it")
        fixed_coco.setdefault("categories", []).append({
            "id": 1,
            "name": "vessel",
            "supercategory": "vessel",
            "keypoints": [],
            "skeleton": []
        })
    
    # Process each annotation
    for i, annotation in enumerate(fixed_coco.get("annotations", [])):
        if "attributes" not in annotation:
            # Add default attributes based on category
            category_id = annotation.get("category_id", 1)
            category_name = "vessel"  # Default
            
            # Find category name based on ID
            for cat in fixed_coco.get("categories", []):
                if cat.get("id") == category_id:
                    category_name = cat.get("name", "vessel")
                    break
            
            # Get default attributes
            default_attrs = get_default_attributes(category_name)
            
            # Set source-specific attributes
            if source == "originaldrsam":
                # Override with original DrSAM specific values
                default_attrs.update({
                    "vessel_id": "SFA",
                    "side": "N/A",
                    "annotator": "human",
                    "label_schema_version": "2025.03"
                })
            elif source == "cvat":
                # CVAT exports might need specific handling
                default_attrs.update({
                    "annotator": "human"
                })
            
            annotation["attributes"] = default_attrs
            errors.append(f"Warning: Added default attributes to annotation {i}")
        else:
            # Validate and fix existing attributes
            category_id = annotation.get("category_id", 1)
            category_name = "vessel"  # Default
            
            # Find category name based on ID
            for cat in fixed_coco.get("categories", []):
                if cat.get("id") == category_id:
                    category_name = cat.get("name", "vessel")
                    break
            
            attr_errors = validate_attributes(category_name, annotation["attributes"])
            if attr_errors:
                for error in attr_errors:
                    errors.append(f"Annotation {i}: {error}")
                
                # Fix attributes
                annotation["attributes"] = fix_attributes(category_name, annotation["attributes"])
                errors.append(f"Fixed attributes for annotation {i}")
    
    return fixed_coco, errors 

def validate_coco_against_schema(coco_data: Dict[str, Any], schema_path: Optional[Union[str, Path]] = None) -> Tuple[bool, List[str]]:
    """
    Validate a COCO format data structure against a JSON schema.
    
    Args:
        coco_data: The COCO format data to validate
        schema_path: Optional path to the schema file. If None, will try to use get_schema_path()
        
    Returns:
        Tuple containing:
        - Boolean indicating if validation passed
        - List of validation error messages
    """
    # Load the schema
    if schema_path is None:
        try:
            schema_path = get_schema_path()
        except FileNotFoundError as e:
            return False, [str(e)]
    
    # Make sure schema_path is a Path object
    schema_path = Path(schema_path)
    
    # Check if schema file exists
    if not schema_path.exists():
        return False, [f"Schema file not found: {schema_path}"]
    
    # Load the JSON schema
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except Exception as e:
        return False, [f"Error loading schema: {e}"]
    
    # Validate the COCO data against the schema
    errors = []
    try:
        jsonschema.validate(instance=coco_data, schema=schema)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        # Get detailed error information
        errors.append(f"Validation error: {e.message}")
        # Add path to the error
        if e.path:
            errors.append(f"Path: {'.'.join(str(p) for p in e.path)}")
        # Add schema path
        if e.schema_path:
            errors.append(f"Schema path: {'.'.join(str(p) for p in e.schema_path)}")
        return False, errors
    except Exception as e:
        errors.append(f"Error during validation: {e}")
        return False, errors 