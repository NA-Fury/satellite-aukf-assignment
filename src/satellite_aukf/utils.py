"""
Utilities for satellite tracking with Orekit and data processing.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import orekit
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from orekit.pyhelpers import setup_orekit_curdir
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, PVCoordinates

logger = logging.getLogger(__name__)


class OrbitPropagator:
    """
    Orbit propagator using Orekit for high-fidelity satellite motion modeling.
    """

    def __init__(self):
        """Initialize Orekit and set up the propagator"""
        # Initialize Orekit data
        try:
            orekit.initVM()
            setup_orekit_curdir()
            _HAVE_OREKIT = True
        except ImportError:
            _HAVE_OREKIT = False
            logging.warning(
                "Orekit not available – falling back to simple ECEF→ECI rotation"
            )

        # Set up time and reference frames
        self.utc = TimeScalesFactory.getUTC()
        self.gcrf = FramesFactory.getGCRF()
        self.itrf = FramesFactory.getITRF(
            orekit.pyhelpers.datetime_to_absolutedate(datetime.utcnow())
        )

        # Earth gravity field (using EGM96)
        self.earth = GravityFieldFactory.getNormalizedProvider(8, 8).getEllipsoid()

        logger.info("Orbit propagator initialized")

    def create_propagator(
        self, initial_state: np.ndarray, epoch: datetime
    ) -> NumericalPropagator:
        """
        Create a numerical propagator with the given initial state.

        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz] in km and km/s
            epoch: Epoch as datetime

        Returns:
            Configured NumericalPropagator
        """
        # Convert to Orekit units (meters)
        pos = Vector3D(
            float(initial_state[0] * 1000),
            float(initial_state[1] * 1000),
            float(initial_state[2] * 1000),
        )
        vel = Vector3D(
            float(initial_state[3] * 1000),
            float(initial_state[4] * 1000),
            float(initial_state[5] * 1000),
        )

        # Create PVCoordinates and orbit
        pv = PVCoordinates(pos, vel)
        orbit_date = AbsoluteDate(
            epoch.year,
            epoch.month,
            epoch.day,
            epoch.hour,
            epoch.minute,
            float(epoch.second),
            self.utc,
        )
        orbit = CartesianOrbit(pv, self.gcrf, orbit_date, Constants.WGS84_EARTH_MU)

        # Create spacecraft state
        sc_state = SpacecraftState(orbit)

        # Set up integrator
        min_step = 0.001
        max_step = 1000.0
        position_tolerance = 1.0
        integrator = DormandPrince853Integrator(
            min_step, max_step, position_tolerance, position_tolerance
        )

        # Create propagator
        propagator = NumericalPropagator(integrator)
        propagator.setInitialState(sc_state)

        # Add force models
        gravity = HolmesFeatherstoneAttractionModel(
            self.gcrf, GravityFieldFactory.getNormalizedProvider(8, 8)
        )
        propagator.addForceModel(gravity)

        return propagator

    def propagate(
        self, initial_state: np.ndarray, epoch: datetime, dt: float
    ) -> np.ndarray:
        """
        Propagate state forward by dt seconds.

        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz] in km and km/s
            epoch: Current epoch
            dt: Time step in seconds

        Returns:
            Propagated state vector
        """
        try:
            propagator = self.create_propagator(initial_state, epoch)

            # Propagate
            target_date = AbsoluteDate(
                epoch.year,
                epoch.month,
                epoch.day,
                epoch.hour,
                epoch.minute,
                float(epoch.second + dt),
                self.utc,
            )
            propagated = propagator.propagate(target_date)

            # Extract state
            pv = propagated.getPVCoordinates(self.gcrf)
            pos = pv.getPosition()
            vel = pv.getVelocity()

            # Convert back to km and km/s
            return np.array(
                [
                    pos.getX() / 1000.0,
                    pos.getY() / 1000.0,
                    pos.getZ() / 1000.0,
                    vel.getX() / 1000.0,
                    vel.getY() / 1000.0,
                    vel.getZ() / 1000.0,
                ]
            )
        except Exception as e:
            logger.error(f"Propagation failed: {e}")
            # Simple two-body propagation as fallback
            return self._two_body_propagation(initial_state, dt)

    def _two_body_propagation(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Simple two-body propagation as fallback"""
        mu = 398600.4418  # km^3/s^2
        r = state[:3]
        v = state[3:]

        # Simple Euler integration
        r_mag = np.linalg.norm(r)
        a = -mu / r_mag**3 * r

        r_new = r + v * dt + 0.5 * a * dt**2
        v_new = v + a * dt

        return np.concatenate([r_new, v_new])


def load_and_preprocess_gnss_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess GNSS measurements.

    Args:
        filepath: Path to GPS measurements parquet file

    Returns:
        Preprocessed DataFrame
    """
    # Load data
    df = pd.read_parquet(filepath)

    # Convert time to datetime
    df["datetime"] = pd.to_datetime(df["time"])

    # Extract position and velocity
    df["x"] = df["ECEF_position"].apply(lambda x: x[0])
    df["y"] = df["ECEF_position"].apply(lambda x: x[1])
    df["z"] = df["ECEF_position"].apply(lambda x: x[2])

    # Convert velocity from dm/s to km/s
    df["vx"] = df["ECEF_velocity"].apply(lambda x: x[0] * 0.0001)
    df["vy"] = df["ECEF_velocity"].apply(lambda x: x[1] * 0.0001)
    df["vz"] = df["ECEF_velocity"].apply(lambda x: x[2] * 0.0001)

    # Filter by date range (May 15-31, 2024)
    df = df[(df["datetime"] >= "2024-05-15") & (df["datetime"] <= "2024-05-31")]

    # Sort by time
    df = df.sort_values("datetime").reset_index(drop=True)

    # Remove obvious outliers (basic filtering)
    r_mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
    valid_mask = (r_mag > 6500) & (r_mag < 8000)  # Reasonable orbit altitudes
    df = df[valid_mask].reset_index(drop=True)

    logger.info(f"Loaded {len(df)} GNSS measurements after preprocessing")

    return df


def measurement_model(state: np.ndarray) -> np.ndarray:
    """
    Measurement model for GNSS observations.
    Maps state to measurement space (identity for direct position/velocity measurements).

    Args:
        state: State vector [x, y, z, vx, vy, vz]

    Returns:
        Expected measurement
    """
    return state  # Direct measurement of full state


def plot_filter_results(results: dict):
    """
    Create comprehensive plots of filter results.

    Args:
        results: Dictionary containing filter results
    """
    times = results["times"]
    states = np.array(results["states"])
    measurements = np.array(results["measurements"])
    covariances = np.array(results["covariances"])
    innovations = np.array(results["innovations"])
    NIS = results["NIS"]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))

    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(
        states[:, 0], states[:, 1], states[:, 2], "b-", label="Filtered", alpha=0.8
    )
    ax1.scatter(
        measurements[::10, 0],
        measurements[::10, 1],
        measurements[::10, 2],
        c="r",
        s=1,
        label="Measurements",
        alpha=0.3,
    )
    ax1.set_xlabel("X (km)")
    ax1.set_ylabel("Y (km)")
    ax1.set_zlabel("Z (km)")
    ax1.set_title("3D Trajectory")
    ax1.legend()

    # Position components
    ax2 = fig.add_subplot(2, 3, 2)
    for i, label in enumerate(["X", "Y", "Z"]):
        ax2.plot(times, states[:, i], label=f"{label} filtered")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Position (km)")
    ax2.set_title("Position Components")
    ax2.legend()
    ax2.grid(True)

    # Velocity components
    ax3 = fig.add_subplot(2, 3, 3)
    for i, label in enumerate(["Vx", "Vy", "Vz"]):
        ax3.plot(times, states[:, i + 3], label=f"{label}")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Velocity (km/s)")
    ax3.set_title("Velocity Components")
    ax3.legend()
    ax3.grid(True)

    # Innovation (residuals)
    ax4 = fig.add_subplot(2, 3, 4)
    for i in range(3):
        ax4.plot(times[1:], innovations[:, i], alpha=0.7)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Innovation (km)")
    ax4.set_title("Position Innovations")
    ax4.grid(True)

    # NIS (Normalized Innovation Squared)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times[1:], NIS, "g-", alpha=0.7)
    ax5.axhline(y=6, color="r", linestyle="--", label="χ² 95% bound")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("NIS")
    ax5.set_title("Normalized Innovation Squared")
    ax5.legend()
    ax5.grid(True)

    # Uncertainty (trace of covariance)
    ax6 = fig.add_subplot(2, 3, 6)
    pos_uncertainty = np.sqrt(np.array([np.trace(P[:3, :3]) for P in covariances]))
    vel_uncertainty = np.sqrt(np.array([np.trace(P[3:, 3:]) for P in covariances]))
    ax6.plot(times, pos_uncertainty, label="Position uncertainty")
    ax6.plot(times, vel_uncertainty, label="Velocity uncertainty")
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Uncertainty (1σ)")
    ax6.set_title("State Uncertainty")
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig("filter_results.png", dpi=300, bbox_inches="tight")
    plt.show()
