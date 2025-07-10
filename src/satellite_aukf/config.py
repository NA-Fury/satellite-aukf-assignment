"""
Configuration module for satellite AUKF package
Defines data paths and settings
"""

import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# Data file paths
RAW_GPS_PATH = DATA_DIR / "GPS_measurements.parquet"
CLEAN_GPS_PATH = DATA_DIR / "GPS_clean.parquet"

# Figures directory
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Orekit data configuration
OREKIT_DATA_PATH = BASE_DIR / "orekit-data"

# Environment variable overrides
if "OREKIT_DATA_PATH" in os.environ:
    OREKIT_DATA_PATH = Path(os.environ["OREKIT_DATA_PATH"])

# Export key paths
__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RAW_GPS_PATH",
    "CLEAN_GPS_PATH",
    "FIGURES_DIR",
    "OREKIT_DATA_PATH",
]
