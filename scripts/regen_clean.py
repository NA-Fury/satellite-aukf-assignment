#!/usr/bin/env python3
"""
regen_clean.py
Regenerate GPS_clean.parquet (and optional ECI columns) from raw GNSS parquet.

Usage examples:
  python regen_clean.py                # uses library-configured data/
  python regen_clean.py --convert-eci  # also add ECI columns
  python regen_clean.py --data-dir D:\alt\data   # override folder
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from satellite_aukf.config import CLEAN_GPS_PATH, RAW_GPS_PATH
from satellite_aukf.utils import OrbitPropagator

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_pivot(raw_path: Path) -> pd.DataFrame:
    """Load long-format GNSS parquet and pivot to wide."""
    df_long = pd.read_parquet(raw_path)
    df_wide = df_long.pivot_table(
        index="datetime", columns="ECEF", values=["position", "velocity"]
    ).reset_index()
    df_wide.columns = [
        "_".join(col).strip() if isinstance(col, tuple) else col
        for col in df_wide.columns
    ]
    # rename 'datetime' column back to a consistent name
    df_wide = df_wide.rename(columns={"datetime_": "datetime"})
    return df_wide


def convert_units(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Convert km→m and dm/s→m/s, return updated df and lists of pos/vel columns."""
    pos_cols = sorted(c for c in df.columns if c.startswith("position_"))
    vel_cols = sorted(c for c in df.columns if c.startswith("velocity_"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df[pos_cols] *= 1000.0  # km → m
    df[vel_cols] /= 10.0  # dm/s → m/s
    return df, pos_cols, vel_cols


def detect_outliers(
    df: pd.DataFrame,
    pos_cols: list[str],
    vel_cols: list[str],
    pos_thr: float,
    vel_thr: float,
) -> pd.DataFrame:
    """Flag rows where |r|>pos_thr or |v|>vel_thr."""
    r = np.linalg.norm(df[pos_cols].values, axis=1)
    v = np.linalg.norm(df[vel_cols].values, axis=1)
    df["is_outlier"] = (r > pos_thr) | (v > vel_thr)
    return df


def interpolate_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Linearly interpolate outlier/missing values in specified cols."""
    df2 = df.sort_values("datetime").set_index("datetime")
    df2[cols] = df2[cols].mask(df2["is_outlier"])
    df2[cols] = df2[cols].interpolate(method="time")
    return df2.reset_index()


def add_eci(df: pd.DataFrame, pos_cols: list[str], vel_cols: list[str]) -> pd.DataFrame:
    """Append eci_position / eci_velocity using Orekit if available."""
    logger.info("Converting ECEF → ECI (Orekit preferred)…")
    prop = OrbitPropagator()  # falls back to simple rotation if no data
    epoch0 = df["datetime"].iloc[0]

    eci_p, eci_v = [], []
    for _, row in df.iterrows():
        p_ecef = row[pos_cols].values
        v_ecef = row[vel_cols].values
        try:
            # OrbitPropagator.propagate expects km / km/s
            state_km = np.concatenate((p_ecef, v_ecef)) / 1000.0
            p_km, v_km = prop.propagate(state_km, row["datetime"])
            p, v = p_km * 1000.0, v_km * 1000.0
        except Exception:
            # final fallback: trivial rotation in metres
            from satellite_aukf.utils import simple_ecef_to_eci

            p, v = simple_ecef_to_eci(p_ecef, v_ecef, row["datetime"], epoch0)
        eci_p.append(p.tolist())
        eci_v.append(v.tolist())

    df["eci_position"] = eci_p
    df["eci_velocity"] = eci_v
    return df


def cli() -> int:
    p = argparse.ArgumentParser(description="Clean raw GNSS parquet")
    p.add_argument("--data-dir", help="override folder containing raw parquet")
    p.add_argument(
        "--position-threshold",
        type=float,
        default=7_000_000,
        help="max orbital radius (m) before flag",
    )
    p.add_argument(
        "--velocity-threshold",
        type=float,
        default=12_000,
        help="max speed (m/s) before flag",
    )
    p.add_argument("--convert-eci", action="store_true", help="add ECI columns")
    args = p.parse_args()

    inp = RAW_GPS_PATH
    out = CLEAN_GPS_PATH
    if args.data_dir:
        d = Path(args.data_dir)
        inp = d / RAW_GPS_PATH.name
        out = d / CLEAN_GPS_PATH.name

    if not inp.exists():
        logger.error("Input parquet not found: %s", inp)
        return 1

    # ── Pipeline ────────────────────────────────────────────────────────────
    logger.info("Loading → %s", inp)
    df = load_and_pivot(inp)
    df, pos_cols, vel_cols = convert_units(df)
    df = detect_outliers(
        df, pos_cols, vel_cols, args.position_threshold, args.velocity_threshold
    )
    n_out = int(df["is_outlier"].sum())
    logger.info("Flagged %d outliers (%.2f%%)", n_out, 100 * n_out / len(df))

    df = interpolate_missing(df, pos_cols + vel_cols)
    if args.convert_eci:
        df = add_eci(df, pos_cols, vel_cols)

    df.attrs.update(
        processed=datetime.utcnow().isoformat(timespec="seconds"),
        outlier_pct=100 * n_out / len(df),
    )
    logger.info("Writing → %s", out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="snappy")
    logger.info("✓ Done – %d rows", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
