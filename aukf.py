"""
Adaptive Unscented Kalman Filter (AUKF) Implementation
Author: Enhanced version for satellite state estimation
Date: 2025

This module implements an Adaptive UKF with multiple noise adaptation methods
for robust satellite tracking using GNSS measurements.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
import scipy.linalg as la
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveMethod(Enum):
    """Enumeration of adaptive filtering methods"""
    SAGE_HUSA = "sage_husa"
    VARIATIONAL_BAYES = "variational_bayes"
    INNOVATION_BASED = "innovation_based"
    MULTIPLE_MODEL = "multiple_model"


@dataclass
class AUKFParameters:
    """Parameters for the Adaptive Unscented Kalman Filter"""
    alpha: float = 1e-3  # Spread of sigma points
    beta: float = 2.0    # Prior knowledge of distribution (2 is optimal for Gaussian)
    kappa: float = 0.0   # Secondary scaling parameter
    adaptive_method: AdaptiveMethod = AdaptiveMethod.SAGE_HUSA
    innovation_window: int = 20  # Window size for innovation-based adaptation
    forgetting_factor: float = 0.98  # For exponential weighting in adaptation
    q_scale_factor: float = 1.0  # Initial scale for process noise
    r_scale_factor: float = 1.0  # Initial scale for measurement noise


class AdaptiveUnscentedKalmanFilter:
    """
    Adaptive Unscented Kalman Filter for satellite state estimation.
    
    This implementation supports multiple adaptive methods for online estimation
    of process and measurement noise covariances.
    """
    
    def __init__(self, 
                 state_dim: int,
                 measurement_dim: int,
                 dt: float,
                 params: Optional[AUKFParameters] = None):
        """
        Initialize the Adaptive Unscented Kalman Filter.
        
        Args:
            state_dim: Dimension of the state vector
            measurement_dim: Dimension of the measurement vector
            dt: Time step
            params: Filter parameters
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.dt = dt
        self.params = params or AUKFParameters()
        
        # State and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # Noise covariances
        self.Q = np.eye(state_dim)
        self.R = np.eye(measurement_dim)
        
        # Sigma point parameters
        self.lambda_ = self.params.alpha**2 * (state_dim + self.params.kappa) - state_dim
        self.gamma = np.sqrt(state_dim + self.lambda_)
        
        # Weights for sigma points
        self.Wm = np.zeros(2 * state_dim + 1)
        self.Wc = np.zeros(2 * state_dim + 1)
        self._compute_weights()
        
        # Adaptive filtering variables
        self.innovation_history = []
        self.S_history = []  # Innovation covariance history
        self.residual_history = []
        self.time_step = 0
        
        # For Sage-Husa adaptation
        self.b_k = 1.0  # Forgetting factor coefficient
        self.d_k = 1.0
        
    def _compute_weights(self):
        """Compute weights for sigma point transformation"""
        n = self.state_dim
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.params.alpha**2 + self.params.beta)
        
        for i in range(1, 2 * n + 1):
            self.Wm[i] = 1.0 / (2 * (n + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (n + self.lambda_))
    
    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for the unscented transformation.
        
        Args:
            x: State vector
            P: State covariance matrix
            
        Returns:
            Sigma points matrix (state_dim x num_sigma_points)
        """
        n = self.state_dim
        sigma_points = np.zeros((n, 2 * n + 1))
        
        # Ensure P is positive definite
        P = 0.5 * (P + P.T)  # Symmetrize
        P += 1e-9 * np.eye(n)  # Add small diagonal for numerical stability
        
        # Compute square root of P using Cholesky decomposition
        try:
            sqrt_P = la.cholesky(P, lower=True)
        except la.LinAlgError:
            # If Cholesky fails, use SVD
            U, s, Vt = la.svd(P)
            sqrt_P = U @ np.diag(np.sqrt(s))
        
        # Generate sigma points
        sigma_points[:, 0] = x
        for i in range(n):
            sigma_points[:, i + 1] = x + self.gamma * sqrt_P[:, i]
            sigma_points[:, n + i + 1] = x - self.gamma * sqrt_P[:, i]
        
        return sigma_points
    
    def predict(self, motion_model, control_input: Optional[np.ndarray] = None):
        """
        Prediction step of the UKF.
        
        Args:
            motion_model: Function that propagates the state forward
            control_input: Optional control input
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points through motion model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(2 * self.state_dim + 1):
            sigma_points_pred[:, i] = motion_model(sigma_points[:, i], self.dt, control_input)
        
        # Compute predicted state
        self.x = np.sum(self.Wm * sigma_points_pred, axis=1)
        
        # Compute predicted covariance
        self.P = np.zeros((self.state_dim, self.state_dim))
        for i in range(2 * self.state_dim + 1):
            y = sigma_points_pred[:, i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)
        
        # Add process noise
        self.P += self.Q
        
        # Store predicted sigma points for update step
        self.sigma_points_pred = sigma_points_pred
    
    def update(self, z: np.ndarray, measurement_model, measurement_jacobian=None):
        """
        Update step of the UKF with adaptive noise estimation.
        
        Args:
            z: Measurement vector
            measurement_model: Function that maps state to measurement
            measurement_jacobian: Optional measurement Jacobian for validation
        """
        # Transform sigma points through measurement model
        n_sigma = 2 * self.state_dim + 1
        sigma_measurements = np.zeros((self.measurement_dim, n_sigma))
        
        for i in range(n_sigma):
            sigma_measurements[:, i] = measurement_model(self.sigma_points_pred[:, i])
        
        # Compute predicted measurement
        z_pred = np.sum(self.Wm * sigma_measurements, axis=1)
        
        # Compute innovation covariance
        S = np.zeros((self.measurement_dim, self.measurement_dim))
        for i in range(n_sigma):
            y = sigma_measurements[:, i] - z_pred
            S += self.Wc[i] * np.outer(y, y)
        S += self.R
        
        # Compute cross-covariance
        Pxz = np.zeros((self.state_dim, self.measurement_dim))
        for i in range(n_sigma):
            dx = self.sigma_points_pred[:, i] - self.x
            dz = sigma_measurements[:, i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)
        
        # Kalman gain
        try:
            K = Pxz @ la.inv(S)
        except la.LinAlgError:
            # Use pseudo-inverse if S is singular
            K = Pxz @ la.pinv(S)
        
        # Innovation
        innovation = z - z_pred
        
        # State update
        self.x += K @ innovation
        
        # Covariance update
        self.P -= K @ S @ K.T
        
        # Ensure positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        self.P += 1e-9 * np.eye(self.state_dim)
        
        # Store innovation for adaptive filtering
        self.innovation_history.append(innovation)
        self.S_history.append(S)
        if len(self.innovation_history) > self.params.innovation_window:
            self.innovation_history.pop(0)
            self.S_history.pop(0)
        
        # Adaptive noise estimation
        self._adapt_noise_covariances(innovation, S, K)
        
        self.time_step += 1
    
    def _adapt_noise_covariances(self, innovation: np.ndarray, S: np.ndarray, K: np.ndarray):
        """
        Adapt process and measurement noise covariances online.
        
        Args:
            innovation: Current innovation vector
            S: Innovation covariance
            K: Kalman gain
        """
        if self.params.adaptive_method == AdaptiveMethod.SAGE_HUSA:
            self._sage_husa_adaptation(innovation, S, K)
        elif self.params.adaptive_method == AdaptiveMethod.INNOVATION_BASED:
            self._innovation_based_adaptation()
        elif self.params.adaptive_method == AdaptiveMethod.VARIATIONAL_BAYES:
            self._variational_bayes_adaptation(innovation, S)
        
    def _sage_husa_adaptation(self, innovation: np.ndarray, S: np.ndarray, K: np.ndarray):
        """Sage-Husa adaptive filtering algorithm"""
        # Update forgetting factors
        self.b_k = self.params.forgetting_factor * self.b_k / (self.b_k + 1)
        self.d_k = (1 - self.b_k) / (1 - self.b_k**(self.time_step + 1))
        
        # Measurement noise adaptation
        C_k = np.outer(innovation, innovation) - S
        self.R = (1 - self.d_k) * self.R + self.d_k * C_k
        
        # Process noise adaptation
        A_k = K @ innovation
        G_k = np.outer(A_k, A_k) + self.P - np.eye(self.state_dim) @ self.P @ np.eye(self.state_dim).T
        self.Q = (1 - self.d_k) * self.Q + self.d_k * G_k
        
        # Ensure positive definiteness
        self.R = 0.5 * (self.R + self.R.T)
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.R += 1e-6 * np.eye(self.measurement_dim)
        self.Q += 1e-6 * np.eye(self.state_dim)
    
    def _innovation_based_adaptation(self):
        """Innovation-based adaptive filtering using windowed covariance estimation"""
        if len(self.innovation_history) < 5:
            return
        
        # Estimate measurement noise from innovation sequence
        innovations = np.array(self.innovation_history)
        S_mean = np.mean(self.S_history, axis=0)
        
        # Empirical innovation covariance
        emp_cov = np.cov(innovations.T)
        
        # Update R to match innovation statistics
        self.R = emp_cov - S_mean + self.R
        self.R = 0.5 * (self.R + self.R.T)
        self.R = np.maximum(self.R, 1e-6 * np.eye(self.measurement_dim))
    
    def _variational_bayes_adaptation(self, innovation: np.ndarray, S: np.ndarray):
        """Simplified variational Bayes adaptation"""
        # Use innovation statistics to adapt noise
        alpha = 0.01  # Learning rate
        
        # Adapt measurement noise
        R_update = np.outer(innovation, innovation) - S + self.R
        self.R = (1 - alpha) * self.R + alpha * R_update
        
        # Simple process noise adaptation
        if hasattr(self, 'prev_innovation'):
            q_scale = np.linalg.norm(innovation - self.prev_innovation) / np.linalg.norm(innovation)
            self.Q *= (1 + alpha * (q_scale - 1))
        
        self.prev_innovation = innovation.copy()
        
        # Ensure positive definiteness
        self.R = 0.5 * (self.R + self.R.T)
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.R = np.maximum(self.R, 1e-6 * np.eye(self.measurement_dim))
        self.Q = np.maximum(self.Q, 1e-6 * np.eye(self.state_dim))
    
    def set_initial_conditions(self, x0: np.ndarray, P0: np.ndarray, 
                              Q0: np.ndarray, R0: np.ndarray):
        """
        Set initial conditions for the filter.
        
        Args:
            x0: Initial state estimate
            P0: Initial state covariance
            Q0: Initial process noise covariance
            R0: Initial measurement noise covariance
        """
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q0.copy() * self.params.q_scale_factor
        self.R = R0.copy() * self.params.r_scale_factor
    
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and covariance.
        
        Returns:
            Tuple of (state estimate, state covariance)
        """
        return self.x.copy(), self.P.copy()
    
    def get_noise_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current noise covariance estimates.
        
        Returns:
            Tuple of (process noise covariance, measurement noise covariance)
        """
        return self.Q.copy(), self.R.copy()
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get filter performance statistics.
        
        Returns:
            Dictionary of filter statistics
        """
        stats = {
            'time_step': self.time_step,
            'state_estimate': self.x.copy(),
            'state_covariance_trace': np.trace(self.P),
            'process_noise_trace': np.trace(self.Q),
            'measurement_noise_trace': np.trace(self.R),
        }
        
        if self.innovation_history:
            innovations = np.array(self.innovation_history)
            stats['innovation_mean'] = np.mean(innovations, axis=0)
            stats['innovation_std'] = np.std(innovations, axis=0)
            stats['normalized_innovation_squared'] = self._compute_nis()
        
        return stats
    
    def _compute_nis(self) -> Optional[float]:
        """Compute Normalized Innovation Squared (NIS) for filter consistency check"""
        if not self.innovation_history or not self.S_history:
            return None
        
        innovation = self.innovation_history[-1]
        S = self.S_history[-1]
        
        try:
            nis = innovation.T @ la.inv(S) @ innovation
            return nis
        except la.LinAlgError:
            return None
    
    def reset_adaptation_history(self):
        """Reset adaptation history for fresh start"""
        self.innovation_history.clear()
        self.S_history.clear()
        self.residual_history.clear()
        self.time_step = 0