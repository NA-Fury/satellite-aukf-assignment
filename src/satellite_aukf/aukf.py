"""
Adaptive Unscented Kalman Filter Implementation for Satellite Tracking

This module implements an Adaptive UKF using the Sage-Husa adaptive algorithm
for online estimation of process and measurement noise covariances.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import cholesky, sqrtm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterParameters:
    """Container for AUKF parameters"""

    alpha: float = 1e-3  # Spread of sigma points
    beta: float = 2.0  # Prior knowledge of distribution (2 is optimal for Gaussian)
    kappa: float = 0.0  # Secondary scaling parameter
    adaptation_window: int = 10  # Window size for noise adaptation
    innovation_gate: float = 3.0  # Chi-squared gate for outlier rejection


class AdaptiveUKF:
    """
    Adaptive Unscented Kalman Filter for satellite state estimation.

    This implementation uses the Sage-Husa adaptive algorithm to estimate
    process and measurement noise covariances online.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        motion_model: callable,
        measurement_model: callable,
        params: Optional[FilterParameters] = None,
    ):
        """
        Initialize the Adaptive UKF.

        Args:
            initial_state: Initial state vector [x, y, z, vx, vy, vz]
            initial_covariance: Initial state covariance matrix (6x6)
            process_noise: Initial process noise covariance Q (6x6)
            measurement_noise: Initial measurement noise covariance R (6x6)
            motion_model: Function that propagates state forward in time
            measurement_model: Function that maps state to measurement space
            params: Filter parameters
        """
        self.x = initial_state.copy()
        self.P = initial_covariance.copy()
        self.Q = process_noise.copy()
        self.R = measurement_noise.copy()
        self.motion_model = motion_model
        self.measurement_model = measurement_model

        self.params = params or FilterParameters()
        self.n = len(initial_state)

        # Calculate sigma point weights
        self._calculate_weights()

        # Initialize adaptation variables
        self.innovation_history = []
        self.residual_history = []
        self.Q_k = self.Q.copy()  # Adaptive Q
        self.R_k = self.R.copy()  # Adaptive R

        # Statistics for filter performance
        self.NIS_history = []  # Normalized Innovation Squared
        self.filter_stats = {"predictions": 0, "updates": 0, "outliers_rejected": 0}

    def _calculate_weights(self):
        """Calculate weights for sigma points"""
        lambda_ = self.params.alpha**2 * (self.n + self.params.kappa) - self.n
        self.lambda_ = lambda_

        # Weights for means
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wm[0] = lambda_ / (self.n + lambda_)
        self.Wm[1:] = 1.0 / (2.0 * (self.n + lambda_))

        # Weights for covariance
        self.Wc = self.Wm.copy()
        self.Wc[0] = self.Wm[0] + (1 - self.params.alpha**2 + self.params.beta)

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for the unscented transform.

        Args:
            x: State vector
            P: Covariance matrix

        Returns:
            Sigma points matrix (n x 2n+1)
        """
        n = len(x)
        sigma_points = np.zeros((n, 2 * n + 1))

        # First sigma point is the mean
        sigma_points[:, 0] = x

        # Calculate square root of (n + lambda) * P
        try:
            sqrt_matrix = cholesky((self.n + self.lambda_) * P)
        except np.linalg.LinAlgError:
            # Fall back to matrix square root if Cholesky fails
            logger.warning("Cholesky decomposition failed, using sqrtm")
            sqrt_matrix = sqrtm((self.n + self.lambda_) * P)

        # Generate remaining sigma points
        for i in range(n):
            sigma_points[:, i + 1] = x + sqrt_matrix[:, i]
            sigma_points[:, i + n + 1] = x - sqrt_matrix[:, i]

        return sigma_points

    def predict(
        self, dt: float, control_input: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of the UKF.

        Args:
            dt: Time step
            control_input: Optional control input

        Returns:
            Predicted state and covariance
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)

        # Propagate sigma points through motion model
        propagated_points = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            propagated_points[:, i] = self.motion_model(sigma_points[:, i], dt)

        # Calculate predicted mean
        self.x_pred = np.sum(self.Wm[:, np.newaxis] * propagated_points, axis=0)

        # Calculate predicted covariance
        self.P_pred = self.Q_k.copy()
        for i in range(2 * self.n + 1):
            y = propagated_points[:, i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(y, y)

        self.filter_stats["predictions"] += 1

        return self.x_pred, self.P_pred

    def update(
        self, measurement: np.ndarray, measurement_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step of the UKF with measurement validation.

        Args:
            measurement: Measurement vector [x, y, z, vx, vy, vz]
            measurement_time: Time of measurement

        Returns:
            Updated state and covariance
        """
        # Generate sigma points from predicted state
        sigma_points = self.generate_sigma_points(self.x_pred, self.P_pred)

        # Transform sigma points to measurement space
        measurement_points = np.zeros((len(measurement), 2 * self.n + 1))
        for i in range(2 * self.n + 1):
            measurement_points[:, i] = self.measurement_model(sigma_points[:, i])

        # Calculate predicted measurement
        z_pred = np.sum(self.Wm[:, np.newaxis] * measurement_points, axis=0)

        # Calculate innovation covariance
        Pzz = self.R_k.copy()
        for i in range(2 * self.n + 1):
            y = measurement_points[:, i] - z_pred
            Pzz += self.Wc[i] * np.outer(y, y)

        # Calculate cross covariance
        Pxz = np.zeros((self.n, len(measurement)))
        for i in range(2 * self.n + 1):
            dx = sigma_points[:, i] - self.x_pred
            dz = measurement_points[:, i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Calculate Kalman gain
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            logger.error("Failed to invert innovation covariance")
            return self.x_pred, self.P_pred

        # Calculate innovation
        innovation = measurement - z_pred

        # Validate measurement (chi-squared test)
        NIS = innovation.T @ np.linalg.inv(Pzz) @ innovation
        self.NIS_history.append(NIS)

        if NIS > self.params.innovation_gate**2 * len(measurement):
            logger.warning(f"Measurement rejected: NIS={NIS:.2f}")
            self.filter_stats["outliers_rejected"] += 1
            return self.x_pred, self.P_pred

        # Update state and covariance
        self.x = self.x_pred + K @ innovation
        self.P = self.P_pred - K @ Pzz @ K.T

        # Store innovation for adaptation
        self.innovation_history.append(innovation)
        if len(self.innovation_history) > self.params.adaptation_window:
            self.innovation_history.pop(0)

        # Update noise estimates
        self._adapt_noise_covariances(K, Pzz, innovation)

        self.filter_stats["updates"] += 1

        return self.x, self.P

    def _adapt_noise_covariances(
        self, K: np.ndarray, S: np.ndarray, innovation: np.ndarray
    ):
        """
        Adapt process and measurement noise covariances using Sage-Husa algorithm.

        Args:
            K: Kalman gain
            S: Innovation covariance
            innovation: Current innovation
        """
        if len(self.innovation_history) < 3:
            return

        # Adaptation gain (decreases over time)
        alpha = 1.0 / (1.0 + self.filter_stats["updates"])

        # Update measurement noise estimate
        C = np.outer(innovation, innovation)
        self.R_k = (1 - alpha) * self.R_k + alpha * (C - S)

        # Ensure positive definiteness
        self.R_k = self._ensure_positive_definite(self.R_k)

        # Update process noise estimate (simplified approach)
        if self.filter_stats["updates"] > 1:
            A = self.P - self.P_pred + K @ S @ K.T
            self.Q_k = (1 - alpha) * self.Q_k + alpha * A
            self.Q_k = self._ensure_positive_definite(self.Q_k)

    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite"""
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def get_state(self) -> Dict[str, Any]:
        """Get current filter state and statistics"""
        return {
            "state": self.x.copy(),
            "covariance": self.P.copy(),
            "Q_adaptive": self.Q_k.copy(),
            "R_adaptive": self.R_k.copy(),
            "statistics": self.filter_stats.copy(),
            "NIS_history": self.NIS_history.copy(),
        }

    def reset(self, state: np.ndarray, covariance: np.ndarray):
        """Reset filter state"""
        self.x = state.copy()
        self.P = covariance.copy()
        self.innovation_history.clear()
        self.NIS_history.clear()
