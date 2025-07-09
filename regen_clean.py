#!/usr/bin/env python3
"""
Regenerate *data/GPS_clean.parquet* from raw GNSS measurements.

Features
--------
* Outlier rejection (position + velocity thresholds)
* Linear interpolation of missing / flagged rows
* Optional ECEF → ECI conversion (Orekit if present, else simple rotation)

Run ``python regen_clean.py --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from utils import CoordinateTransforms, DataProcessor, OrekitInitializer

# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> int:
    """CLI entry point."""
    data_dir = Path(args.data_dir)
    inp = data_dir / "GPS_measurements.parquet"
    out = data_dir / "GPS_clean.parquet"

    if not inp.exists():
        logger.error("Input file not found: %s", inp)
        return 1

    logger.info("Loading raw GNSS data …")
    gps_df = DataProcessor.load_gps_data(str(inp))
    logger.info("Loaded %,d rows", len(gps_df))

    # ── detect & interpolate outliers ───────────────────────────────────────
    gps_df = DataProcessor.detect_outliers(
        gps_df,
        position_threshold=args.position_threshold,
        velocity_threshold=args.velocity_threshold,
    )
    n_out = int(gps_df["is_outlier"].sum())
    logger.info("Flagged %,d outliers (%.2f %%)", n_out, n_out / len(gps_df) * 100)

    gps_clean = DataProcessor.interpolate_missing_data(gps_df)

    # ── optional ECEF → ECI --------------------------------------------------
    if args.convert_eci:
        _add_eci_columns(gps_clean)

    # ── metadata & write parquet ────────────────────────────────────────────
    gps_clean.attrs.update(
        processed_date=datetime.utcnow().isoformat(timespec="seconds"),
        outlier_count=n_out,
        outlier_percentage=n_out / len(gps_df) * 100,
    )
    logger.info("Writing cleaned data → %s", out)
    gps_clean.to_parquet(out, compression="snappy")

    logger.info(
        "✓ Done – %,d rows • %d SV • %s … %s",
        len(gps_clean),
        gps_clean["sv"].nunique(),
        gps_clean["datetime"].min(),
        gps_clean["datetime"].max(),
    )
    return 0


# ---------------------------------------------------------------------------


def _add_eci_columns(df) -> None:  # noqa: ANN001  (keep signature simple)
    """Append ``eci_position`` & ``eci_velocity`` list columns in-place."""
    logger.info("Converting ECEF → ECI … (Orekit preferred)")

    try:
        OrekitInitializer.initialize()
        use_orekit = True
    except Exception:
        logger.warning("Orekit unavailable – falling back to simple rotation")
        use_orekit = False

    eci_pos, eci_vel = [], []
    start_time = df["datetime"].iloc[0]

    for idx, row in df.iterrows():
        if idx % 1_000 == 0:
            logger.info("  progress %d / %d", idx, len(df))

        ecef_p = np.array([row[f"{ax}_ecef"] for ax in ("x", "y", "z")])
        ecef_v = np.array([row[f"v{ax}_ecef"] for ax in ("x", "y", "z")])

        if use_orekit:
            try:
                p, v = CoordinateTransforms.ecef_to_eci(ecef_p, ecef_v, row["datetime"])
            except Exception:
                p, v = simple_ecef_to_eci(ecef_p, ecef_v, row["datetime"], start_time)
        else:
            p, v = simple_ecef_to_eci(ecef_p, ecef_v, row["datetime"], start_time)

        eci_pos.append(p.tolist())
        eci_vel.append(v.tolist())

    df["eci_position"] = eci_pos
    df["eci_velocity"] = eci_vel


def simple_ecef_to_eci(
    ecef_pos: NDArray[np.float64],
    ecef_vel: NDArray[np.float64],
    current: datetime,
    epoch0: datetime,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ultra-light ECEF→ECI rotation (ignores polar motion, precession, nutation)."""
    omega = 7.292_115_9e-5  # rad s⁻¹
    theta = omega * (current - epoch0).total_seconds()

    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    r_eci = rot @ ecef_pos
    v_eci = rot @ ecef_vel + np.cross([0.0, 0.0, omega], r_eci)
    return r_eci, v_eci


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean & optionally rotate GNSS parquet")
    p.add_argument(
        "--data-dir", default="data", help="directory containing raw parquet"
    )
    p.add_argument("--position-threshold", type=float, default=50_000, help="metres")
    p.add_argument("--velocity-threshold", type=float, default=1_000, help="m s⁻¹")
    p.add_argument("--convert-eci", action="store_true", help="add ECI columns")
    raise SystemExit(main(p.parse_args()))
