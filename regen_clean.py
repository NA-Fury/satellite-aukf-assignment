#!/usr/bin/env python3
"""
Regenerate clean GPS data from raw measurements.

This script processes the raw GPS_measurements.parquet file to create
a cleaned version with outlier detection, interpolation, and coordinate
transformations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import argparse

# Import utilities
from utils import DataProcessor, CoordinateTransforms, OrekitInitializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main processing function."""
    
    # Define paths
    data_dir = Path(args.data_dir)
    input_file = data_dir / "GPS_measurements.parquet"
    output_file = data_dir / "GPS_clean.parquet"
    
    # Check input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1
    
    logger.info(f"Loading raw GPS data from {input_file}")
    
    # Load raw data
    gps_data = DataProcessor.load_gps_data(str(input_file))
    logger.info(f"Loaded {len(gps_data):,} measurements")
    
    # Detect outliers
    logger.info("Detecting outliers...")
    gps_data = DataProcessor.detect_outliers(
        gps_data,
        position_threshold=args.position_threshold,
        velocity_threshold=args.velocity_threshold
    )
    
    n_outliers = gps_data['is_outlier'].sum()
    logger.info(f"Found {n_outliers:,} outliers ({n_outliers/len(gps_data)*100:.2f}%)")
    
    # Interpolate missing data
    logger.info("Interpolating missing/outlier data...")
    gps_clean = DataProcessor.interpolate_missing_data(gps_data)
    
    # Convert to ECI coordinates if requested
    if args.convert_eci:
        logger.info("Converting ECEF to ECI coordinates...")
        
        # Initialize Orekit if available
        try:
            OrekitInitializer.initialize()
            use_orekit = True
        except:
            logger.warning("Orekit not available, using simplified transformation")
            use_orekit = False
        
        eci_positions = []
        eci_velocities = []
        
        for idx, row in gps_clean.iterrows():
            if idx % 1000 == 0:
                logger.info(f"  Processing {idx}/{len(gps_clean)}...")
            
            ecef_pos = np.array([row['x_ecef'], row['y_ecef'], row['z_ecef']])
            ecef_vel = np.array([row['vx_ecef'], row['vy_ecef'], row['vz_ecef']])
            
            if use_orekit:
                try:
                    eci_pos, eci_vel = CoordinateTransforms.ecef_to_eci(
                        ecef_pos, ecef_vel, row['datetime']
                    )
                except:
                    # Fallback to simple rotation
                    eci_pos, eci_vel = simple_ecef_to_eci(
                        ecef_pos, ecef_vel, row['datetime'], gps_clean['datetime'].iloc[0]
                    )
            else:
                eci_pos, eci_vel = simple_ecef_to_eci(
                    ecef_pos, ecef_vel, row['datetime'], gps_clean['datetime'].iloc[0]
                )
            
            eci_positions.append(eci_pos.tolist())
            eci_velocities.append(eci_vel.tolist())
        
        gps_clean['eci_position'] = eci_positions
        gps_clean['eci_velocity'] = eci_velocities
    
    # Add metadata
    gps_clean.attrs['processed_date'] = datetime.now().isoformat()
    gps_clean.attrs['outlier_count'] = int(n_outliers)
    gps_clean.attrs['outlier_percentage'] = float(n_outliers/len(gps_data)*100)
    
    # Save cleaned data
    logger.info(f"Saving cleaned data to {output_file}")
    gps_clean.to_parquet(output_file, compression='snappy')
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total measurements: {len(gps_clean):,}")
    logger.info(f"  Satellites: {gps_clean['sv'].nunique()}")
    logger.info(f"  Time range: {gps_clean['datetime'].min()} to {gps_clean['datetime'].max()}")
    logger.info(f"  Position range (km): X=[{gps_clean['x_ecef'].min()/1000:.1f}, {gps_clean['x_ecef'].max()/1000:.1f}]")
    
    logger.info("\n✓ Data cleaning complete!")
    return 0


def simple_ecef_to_eci(ecef_pos, ecef_vel, current_time, start_time):
    """Simple ECEF to ECI transformation using Earth rotation."""
    # Earth rotation rate
    omega = 7.2921159e-5  # rad/s
    
    # Time since start
    dt = (current_time - start_time).total_seconds()
    
    # Rotation angle
    theta = omega * dt
    
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Transform position and velocity
    eci_pos = R @ ecef_pos
    eci_vel = R @ ecef_vel + np.cross([0, 0, omega], eci_pos)
    
    return eci_pos, eci_vel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate clean GPS data from raw measurements"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing GPS data files"
    )
    parser.add_argument(
        "--position-threshold",
        type=float,
        default=50000,
        help="Position outlier threshold in meters"
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=1000,
        help="Velocity outlier threshold in m/s"
    )
    parser.add_argument(
        "--convert-eci",
        action="store_true",
        help="Convert ECEF to ECI coordinates"
    )
    
    args = parser.parse_args()
    exit(main(args))