import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import orekit
from org.orekit.frames import FramesFactory
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
import warnings


class OrekitInitializer:
    """Initialize Orekit with proper data context"""
    
    @staticmethod
    def initialize():
        """Initialize Orekit VM and data context"""
        try:
            orekit.initVM()
            # You'll need to set up orekit-data directory
            # Download from: https://gitlab.orekit.org/orekit/orekit-data
            from orekit.pyhelpers import setup_orekit_curdir
            setup_orekit_curdir()
        except Exception as e:
            warnings.warn(f"Orekit initialization warning: {e}")


class CoordinateTransforms:
    """Coordinate transformation utilities"""
    
    @staticmethod
    def ecef_to_eci(ecef_pos: np.ndarray, ecef_vel: np.ndarray, 
                    epoch: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ECEF coordinates to ECI (J2000).
        
        Args:
            ecef_pos: Position in ECEF frame (meters)
            ecef_vel: Velocity in ECEF frame (m/s)
            epoch: Time of the state
            
        Returns:
            Tuple of (ECI position, ECI velocity)
        """
        # Get frames
        ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        eci = FramesFactory.getEME2000()
        
        # Convert epoch to Orekit AbsoluteDate
        utc = TimeScalesFactory.getUTC()
        date = AbsoluteDate(epoch.year, epoch.month, epoch.day,
                           epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6,
                           utc)
        
        # Create PVCoordinates in ECEF
        pv_ecef = PVCoordinates(
            orekit.Vector3D(float(ecef_pos[0]), float(ecef_pos[1]), float(ecef_pos[2])),
            orekit.Vector3D(float(ecef_vel[0]), float(ecef_vel[1]), float(ecef_vel[2]))
        )
        
        # Transform to ECI
        transform = ecef.getTransformTo(eci, date)
        pv_eci = transform.transformPVCoordinates(pv_ecef)
        
        # Extract numpy arrays
        pos_eci = np.array([pv_eci.getPosition().getX(),
                           pv_eci.getPosition().getY(),
                           pv_eci.getPosition().getZ()])
        vel_eci = np.array([pv_eci.getVelocity().getX(),
                           pv_eci.getVelocity().getY(),
                           pv_eci.getVelocity().getZ()])
        
        return pos_eci, vel_eci
    
    @staticmethod
    def eci_to_ecef(eci_pos: np.ndarray, eci_vel: np.ndarray,
                    epoch: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ECI (J2000) coordinates to ECEF.
        
        Args:
            eci_pos: Position in ECI frame (meters)
            eci_vel: Velocity in ECI frame (m/s)
            epoch: Time of the state
            
        Returns:
            Tuple of (ECEF position, ECEF velocity)
        """
        # Get frames
        ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        eci = FramesFactory.getEME2000()
        
        # Convert epoch to Orekit AbsoluteDate
        utc = TimeScalesFactory.getUTC()
        date = AbsoluteDate(epoch.year, epoch.month, epoch.day,
                           epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6,
                           utc)
        
        # Create PVCoordinates in ECI
        pv_eci = PVCoordinates(
            orekit.Vector3D(float(eci_pos[0]), float(eci_pos[1]), float(eci_pos[2])),
            orekit.Vector3D(float(eci_vel[0]), float(eci_vel[1]), float(eci_vel[2]))
        )
        
        # Transform to ECEF
        transform = eci.getTransformTo(ecef, date)
        pv_ecef = transform.transformPVCoordinates(pv_eci)
        
        # Extract numpy arrays
        pos_ecef = np.array([pv_ecef.getPosition().getX(),
                            pv_ecef.getPosition().getY(),
                            pv_ecef.getPosition().getZ()])
        vel_ecef = np.array([pv_ecef.getVelocity().getX(),
                            pv_ecef.getVelocity().getY(),
                            pv_ecef.getVelocity().getZ()])
        
        return pos_ecef, vel_ecef


class OrbitPropagator:
    """High-fidelity orbit propagator using Orekit"""
    
    def __init__(self, use_high_fidelity: bool = True, 
                 gravity_degree: int = 20, gravity_order: int = 20):
        """
        Initialize orbit propagator.
        
        Args:
            use_high_fidelity: Use numerical propagator with perturbations
            gravity_degree: Degree of gravity field
            gravity_order: Order of gravity field
        """
        self.use_high_fidelity = use_high_fidelity
        self.gravity_degree = gravity_degree
        self.gravity_order = gravity_order
        
        # Initialize Orekit if not already done
        OrekitInitializer.initialize()
        
        # Reference frames
        self.eci_frame = FramesFactory.getEME2000()
        self.ecef_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        
        # Earth model
        self.earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            self.ecef_frame
        )
        
        # Time scale
        self.utc = TimeScalesFactory.getUTC()
    
    def propagate_state(self, initial_state: np.ndarray, dt: float,
                       epoch: datetime, satellite_properties: Optional[Dict] = None) -> np.ndarray:
        """
        Propagate satellite state forward in time.
        
        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz] in ECI (meters and m/s)
            dt: Time step in seconds
            epoch: Epoch of initial state
            satellite_properties: Optional dict with 'mass', 'drag_area', 'drag_coeff', 'srp_area', 'srp_coeff'
            
        Returns:
            Propagated state vector in ECI
        """
        # Convert to Orekit date
        initial_date = AbsoluteDate(
            epoch.year, epoch.month, epoch.day,
            epoch.hour, epoch.minute, epoch.second + epoch.microsecond/1e6,
            self.utc
        )
        
        # Create initial orbit
        initial_pv = PVCoordinates(
            orekit.Vector3D(float(initial_state[0]), float(initial_state[1]), float(initial_state[2])),
            orekit.Vector3D(float(initial_state[3]), float(initial_state[4]), float(initial_state[5]))
        )
        
        initial_orbit = CartesianOrbit(
            initial_pv, self.eci_frame, initial_date, Constants.WGS84_EARTH_MU
        )
        
        # Create spacecraft state
        spacecraft_state = SpacecraftState(initial_orbit)
        
        if satellite_properties:
            mass = satellite_properties.get('mass', 500.0)  # Default 500 kg
            spacecraft_state = SpacecraftState(initial_orbit, mass)
        
        # Create propagator
        if self.use_high_fidelity:
            propagator = self._create_numerical_propagator(
                spacecraft_state, satellite_properties
            )
        else:
            propagator = KeplerianPropagator(initial_orbit)
        
        # Propagate
        target_date = initial_date.shiftedBy(dt)
        propagated_state = propagator.propagate(target_date)
        
        # Extract state vector
        pv = propagated_state.getPVCoordinates()
        state_vector = np.array([
            pv.getPosition().getX(),
            pv.getPosition().getY(),
            pv.getPosition().getZ(),
            pv.getVelocity().getX(),
            pv.getVelocity().getY(),
            pv.getVelocity().getZ()
        ])
        
        return state_vector
    
    def _create_numerical_propagator(self, initial_state: SpacecraftState,
                                   satellite_properties: Optional[Dict] = None) -> NumericalPropagator:
        """Create high-fidelity numerical propagator with perturbations"""
        
        # Adaptive step integrator
        min_step = 0.001
        max_step = 1000.0
        position_tolerance = 1.0  # meters
        
        integrator = DormandPrince853Integrator(
            min_step, max_step, position_tolerance, position_tolerance
        )
        
        propagator = NumericalPropagator(integrator)
        propagator.setInitialState(initial_state)
        
        # Add gravity field (high degree and order)
        gravity_provider = GravityFieldFactory.getNormalizedProvider(
            self.gravity_degree, self.gravity_order
        )
        gravity = HolmesFeatherstoneAttractionModel(
            self.ecef_frame, gravity_provider
        )
        propagator.addForceModel(gravity)
        
        # Add atmospheric drag if properties provided
        if satellite_properties:
            drag_area = satellite_properties.get('drag_area', 10.0)  # m²
            drag_coeff = satellite_properties.get('drag_coeff', 2.2)
            
            sun = CelestialBodyFactory.getSun()
            atmosphere = NRLMSISE00(
                NRLMSISE00.InputParams(), sun, self.earth
            )
            
            drag = DragForce(
                atmosphere,
                IsotropicDrag(drag_area, drag_coeff)
            )
            propagator.addForceModel(drag)
            
            # Add solar radiation pressure
            srp_area = satellite_properties.get('srp_area', drag_area)
            srp_coeff = satellite_properties.get('srp_coeff', 1.5)
            
            srp = SolarRadiationPressure(
                sun, Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                IsotropicRadiationSingleCoefficient(srp_area, srp_coeff)
            )
            propagator.addForceModel(srp)
        
        return propagator


class DataProcessor:
    """Process GNSS measurement data"""
    
    @staticmethod
    def load_gps_data(filepath: str) -> pd.DataFrame:
        """
        Load GPS measurements from parquet file.
        
        Args:
            filepath: Path to GPS measurements file
            
        Returns:
            Processed DataFrame with measurements
        """
        df = pd.read_parquet(filepath)
        
        # Convert time to datetime
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Extract ECEF positions (convert from km to m)
        df['x_ecef'] = df['ECEF'].apply(lambda x: x['position'][0] * 1000)
        df['y_ecef'] = df['ECEF'].apply(lambda x: x['position'][1] * 1000)
        df['z_ecef'] = df['ECEF'].apply(lambda x: x['position'][2] * 1000)
        
        # Extract velocities (convert from dm/s to m/s)
        df['vx_ecef'] = df['velocity'].apply(lambda x: x[0] / 10.0)
        df['vy_ecef'] = df['velocity'].apply(lambda x: x[1] / 10.0)
        df['vz_ecef'] = df['velocity'].apply(lambda x: x[2] / 10.0)
        
        return df
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, 
                       position_threshold: float = 50000,  # 50 km
                       velocity_threshold: float = 1000) -> pd.DataFrame:  # 1 km/s
        """
        Detect and mark outliers in GPS data.
        
        Args:
            data: GPS measurement DataFrame
            position_threshold: Maximum position change between measurements (m)
            velocity_threshold: Maximum velocity change between measurements (m/s)
            
        Returns:
            DataFrame with outlier flags
        """
        data = data.copy()
        
        # Calculate differences
        pos_cols = ['x_ecef', 'y_ecef', 'z_ecef']
        vel_cols = ['vx_ecef', 'vy_ecef', 'vz_ecef']
        
        # Position differences
        pos_diff = np.sqrt(sum((data[col].diff()**2 for col in pos_cols)))
        
        # Velocity differences
        vel_diff = np.sqrt(sum((data[col].diff()**2 for col in vel_cols)))
        
        # Mark outliers
        data['is_outlier'] = (pos_diff > position_threshold) | (vel_diff > velocity_threshold)
        
        # Also check for NaN values
        data['is_outlier'] |= data[pos_cols + vel_cols].isnull().any(axis=1)
        
        return data
    
    @staticmethod
    def interpolate_missing_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing or outlier data points.
        
        Args:
            data: GPS measurement DataFrame with outlier flags
            
        Returns:
            DataFrame with interpolated values
        """
        data = data.copy()
        
        # Columns to interpolate
        cols_to_interp = ['x_ecef', 'y_ecef', 'z_ecef', 
                         'vx_ecef', 'vy_ecef', 'vz_ecef']
        
        # Replace outliers with NaN
        for col in cols_to_interp:
            data.loc[data['is_outlier'], col] = np.nan
        
        # Interpolate
        data[cols_to_interp] = data[cols_to_interp].interpolate(method='cubic')
        
        # Fill any remaining NaN values
        data[cols_to_interp] = data[cols_to_interp].fillna(method='bfill').fillna(method='ffill')
        
        return data


class FilterTuner:
    """Utilities for tuning filter parameters"""
    
    @staticmethod
    def estimate_initial_covariance(measurements: pd.DataFrame) -> np.ndarray:
        """
        Estimate initial state covariance from measurements.
        
        Args:
            measurements: DataFrame with GPS measurements
            
        Returns:
            Initial covariance matrix P0
        """
        # Extract position and velocity columns
        pos_cols = ['x_ecef', 'y_ecef', 'z_ecef']
        vel_cols = ['vx_ecef', 'vy_ecef', 'vz_ecef']
        
        # Compute standard deviations
        pos_std = measurements[pos_cols].std().values
        vel_std = measurements[vel_cols].std().values
        
        # Create diagonal covariance matrix
        P0 = np.diag(np.concatenate([pos_std**2, vel_std**2]))
        
        # Scale based on measurement quality
        P0 *= 10  # Conservative initial uncertainty
        
        return P0
    
    @staticmethod
    def estimate_process_noise(dt: float, 
                             acceleration_std: float = 0.1) -> np.ndarray:
        """
        Estimate process noise covariance matrix.
        
        Args:
            dt: Time step
            acceleration_std: Standard deviation of acceleration (m/s²)
            
        Returns:
            Process noise covariance matrix Q
        """
        # Continuous white noise acceleration model
        q = acceleration_std**2
        
        # Discrete process noise for position-velocity model
        Q = np.array([
            [dt**4/4, 0, 0, dt**3/2, 0, 0],
            [0, dt**4/4, 0, 0, dt**3/2, 0],
            [0, 0, dt**4/4, 0, 0, dt**3/2],
            [dt**3/2, 0, 0, dt**2, 0, 0],
            [0, dt**3/2, 0, 0, dt**2, 0],
            [0, 0, dt**3/2, 0, 0, dt**2]
        ]) * q
        
        return Q
    
    @staticmethod
    def estimate_measurement_noise(measurements: pd.DataFrame,
                                 window_size: int = 100) -> np.ndarray:
        """
        Estimate measurement noise from data.
        
        Args:
            measurements: DataFrame with GPS measurements
            window_size: Window size for noise estimation
            
        Returns:
            Measurement noise covariance matrix R
        """
        # Use innovation-based approach
        pos_cols = ['x_ecef', 'y_ecef', 'z_ecef']
        vel_cols = ['vx_ecef', 'vy_ecef', 'vz_ecef']
        
        # Compute measurement differences
        meas_diff = pd.DataFrame()
        for col in pos_cols + vel_cols:
            meas_diff[col] = measurements[col].diff()
        
        # Remove outliers using IQR
        Q1 = meas_diff.quantile(0.25)
        Q3 = meas_diff.quantile(0.75)
        IQR = Q3 - Q1
        
        mask = ~((meas_diff < (Q1 - 1.5 * IQR)) | (meas_diff > (Q3 + 1.5 * IQR))).any(axis=1)
        meas_diff_clean = meas_diff[mask]
        
        # Estimate noise covariance
        if len(meas_diff_clean) > window_size:
            noise_cov = meas_diff_clean.cov().values
        else:
            # Fallback to simple diagonal estimate
            noise_std = meas_diff_clean.std().values
            noise_cov = np.diag(noise_std**2)
        
        # Scale and ensure positive definiteness
        R = 0.5 * (noise_cov + noise_cov.T)
        R += 1e-6 * np.eye(6)
        
        return R