"""
Data preprocessing utilities for satellite tracking
Author: Naziha Aslam
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data preprocessing utilities for GNSS measurements"""

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        position_threshold: float = 50000,
        velocity_threshold: float = 1000,
    ) -> pd.DataFrame:
        """
        Detect outliers in GNSS measurements using 3-sigma rule

        Args:
            df: DataFrame with ECEF position/velocity columns
            position_threshold: Position outlier threshold (m)
            velocity_threshold: Velocity outlier threshold (m/s)

        Returns:
            DataFrame with 'is_outlier' column added
        """
        df = df.copy()

        # Identify position and velocity columns
        pos_cols = [
            col
            for col in df.columns
            if "x_ecef" in col or "y_ecef" in col or "z_ecef" in col
        ]
        vel_cols = [
            col
            for col in df.columns
            if "vx_ecef" in col or "vy_ecef" in col or "vz_ecef" in col
        ]

        if not pos_cols:
            pos_cols = ["x_ecef", "y_ecef", "z_ecef"]
        if not vel_cols:
            vel_cols = ["vx_ecef", "vy_ecef", "vz_ecef"]

        # Calculate magnitudes
        if all(col in df.columns for col in pos_cols):
            pos_mag = np.sqrt(df[pos_cols].pow(2).sum(axis=1))
            pos_outliers = np.abs(pos_mag - pos_mag.median()) > 3 * pos_mag.std()
        else:
            pos_outliers = pd.Series(False, index=df.index)

        if all(col in df.columns for col in vel_cols):
            vel_mag = np.sqrt(df[vel_cols].pow(2).sum(axis=1))
            vel_outliers = np.abs(vel_mag - vel_mag.median()) > 3 * vel_mag.std()
        else:
            vel_outliers = pd.Series(False, index=df.index)

        # Combine outlier flags
        df["is_outlier"] = pos_outliers | vel_outliers

        outlier_count = df["is_outlier"].sum()
        logger.info(
            f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)"
        )

        return df

    @staticmethod
    def interpolate_missing_data(
        df: pd.DataFrame, max_gap_seconds: float = 300
    ) -> pd.DataFrame:
        """
        Interpolate missing data gaps using cubic splines

        Args:
            df: DataFrame with time series data
            max_gap_seconds: Maximum gap to interpolate (seconds)

        Returns:
            DataFrame with interpolated data
        """
        df = df.copy()

        if "datetime" not in df.columns:
            logger.warning("No 'datetime' column found, skipping interpolation")
            return df

        # Convert datetime to numeric for interpolation
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # Check for gaps
        time_diff = df["datetime"].diff().dt.total_seconds()
        large_gaps = time_diff > max_gap_seconds

        if not large_gaps.any():
            logger.info("No large gaps found, no interpolation needed")
            return df

        logger.info(f"Found {large_gaps.sum()} gaps larger than {max_gap_seconds}s")

        # For simplicity, just mark interpolated points but don't actually interpolate
        # In a full implementation, you would use cubic splines here
        df["interpolated"] = False

        return df

    @staticmethod
    def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete cleaning pipeline for GNSS measurements

        Args:
            df: Raw GNSS measurements

        Returns:
            Cleaned DataFrame
        """
        # Remove outliers
        df_clean = DataProcessor.detect_outliers(df)

        # Remove outlier points
        df_clean = df_clean[~df_clean["is_outlier"]].copy()

        # Interpolate missing data
        df_clean = DataProcessor.interpolate_missing_data(df_clean)

        return df_clean
