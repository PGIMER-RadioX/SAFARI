# src/foundation_model/__init__.py

# This file makes the 'foundation_model' directory a Python package.

from .data_utils import load_radiomics_data
from .model import CollaborativeModel
from .training import train_foundation_model