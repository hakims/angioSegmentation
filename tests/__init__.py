"""
Dr-SAM Testing Framework

This package contains tools for testing the Dr-SAM pipeline,
including validation against the original Dr-SAM dataset.
"""

# Import main components for convenience
from tests.originalDrSAM import DrSAMValidator, run_drsam_pipeline

__all__ = [
    'DrSAMValidator',
    'run_drsam_pipeline'
] 