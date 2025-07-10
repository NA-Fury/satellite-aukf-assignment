"""
Enhanced Adaptive Unscented Kalman Filter (AUKF) for Satellite Tracking
Author: Naziha Aslam
License: MIT
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky, solve_triangular, svd
from scipy.stats import chi2

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Enums & parameters
# ────────────────────────────────────────────────────────────────────────────
class AdaptiveMethod(Enum):
    NONE = "none"
    SAGE_HUSA = "sage_husa"
    INNOVATION_BASED = "innovation_based"
    VARIATIONAL_BAYES = "variational_bayes"


@dataclass
class AUKFParameters:
    alpha: float = 0.01
    beta: float = 2.0
    kappa: float = 3.0
    adaptive_method: AdaptiveMethod = AdaptiveMethod.SAGE_HUSA
    innovation_window: int = 10
    forgetting_factor: float = 0.92


# ────────────────────────────────────────────────────────────────────────────
# Adaptive Unscented Kalman Filter
# ────────────────────────────────────────────────────────────────────────────
class AdaptiveUKF:
    """Minimal‑yet‑robust adaptive Unscented Kalman Filter."""

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dt: float,
        *,
        fx: Callable[[np.ndarray, float], np.ndarray],
        hx: Callable[[np.ndarray], np.ndarray],
        params: Optional[AUKFParameters] = None,
    ) -> None:
        # dimensions / callbacks --------------------------------------------------
        self.n = self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx, self.hx = fx, hx
        self.params = params or AUKFParameters()

        # primary state & covariances --------------------------------------------
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.zeros((dim_x, dim_x))
        self.R = np.eye(dim_z)
        self.Q_adaptive = self.Q.copy()
        self.R_adaptive = self.R.copy()

        # UT weights --------------------------------------------------------------
        self.lambda_ = self.params.alpha**2 * (self.n + self.params.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        self._compute_weights()

        # histories ---------------------------------------------------------------
        self.innovation_history: List[np.ndarray] = []
        self.nis_history: List[float] = []
        self.condition_numbers: List[float] = []

        # Store previous state for Sage-Husa adaptation
        self.x_prior = None

    def _compute_weights(self) -> None:
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.params.alpha**2 + self.params.beta)

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Return sigma‑point matrix *exactly* centered at ``x``."""
        x = x.astype(float)
        try:
            U = np.linalg.cholesky((self.n + self.lambda_) * P)
        except LinAlgError:
            # SVD fallback (handles semi‑definite P)
            U_s, S, _ = np.linalg.svd(P, full_matrices=False)
            U = np.sqrt(self.n + self.lambda_) * (U_s * np.sqrt(np.maximum(S, 1e-12)))

        sigma = np.empty((self.n, 2 * self.n + 1))
        sigma[:, 0] = x
        sigma[:, 1 : self.n + 1] = x[:, None] + U
        sigma[:, self.n + 1 :] = x[:, None] - U

        # exact‑mean enforcement (shift all points by residual)
        resid = x - sigma @ self.Wm
        sigma += resid[:, None]
        return sigma

    def predict(self) -> None:
        # Store conditioning before time update
        self.condition_numbers.append(float(np.linalg.cond(self.P)))

        # Store current state for adaptation
        self.x_prior = self.x.copy()

        # Generate sigma points from current state
        S = self.generate_sigma_points(self.x, self.P)
        S_pred = np.column_stack([self.fx(S[:, i], self.dt) for i in range(S.shape[1])])

        # Compute predicted mean and store for Sage-Husa
        self.x_predicted = S_pred @ self.Wm  # ADDED: Store prediction
        self.x = self.x_predicted.copy()

        # Compute predicted covariance
        self.P = self.Q_adaptive.copy()
        for i in range(S_pred.shape[1]):
            d = S_pred[:, i] - self.x
            self.P += self.Wc[i] * np.outer(d, d)
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray) -> None:
        S = self.generate_sigma_points(self.x, self.P)
        Z = np.column_stack([self.hx(S[:, i]) for i in range(S.shape[1])])
        z_pred = Z @ self.Wm

        Pzz = self.R_adaptive.copy()
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(Z.shape[1]):
            dz = Z[:, i] - z_pred
            dx = S[:, i] - self.x
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        v = z - z_pred
        self.x += K @ v
        self.P -= K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # stats
        self.innovation_history.append(v)
        try:
            self.nis_history.append(float(v.T @ np.linalg.inv(Pzz) @ v))
        except LinAlgError:
            try:
                self.nis_history.append(float(v.T @ np.linalg.pinv(Pzz) @ v))
            except:
                pass

        self._adapt_noise_covariances()

    def _adapt_noise_covariances(self) -> None:
        method = self.params.adaptive_method
        if method is AdaptiveMethod.NONE:
            return
        if method is AdaptiveMethod.SAGE_HUSA:
            self._sage_husa()
        elif method is AdaptiveMethod.INNOVATION_BASED:
            self._innovation_based()
        elif method is AdaptiveMethod.VARIATIONAL_BAYES:
            self._variational_bayes()

    def _sage_husa(self) -> None:
        """
        Sage-Husa
        """
        if not self.innovation_history or self.x_prior is None:
            return

        rho = self.params.forgetting_factor
        v = self.innovation_history[-1]  # Innovation (measurement residual)

        # FIXED: Proper measurement noise adaptation using innovation covariance
        # This is the mathematically correct approach
        self.R_adaptive = rho * self.R_adaptive + (1 - rho) * np.outer(v, v)

        # FIXED: Process noise adaptation using state prediction residual
        # This should be the residual between predicted and actual state estimates
        if hasattr(self, "x_predicted") and self.x_predicted is not None:
            # Use the prediction residual from the predict step
            dx_process = self.x - self.x_predicted
        else:
            # Fallback: use velocity-based process noise estimate
            # This is more appropriate for orbital mechanics
            vel_magnitude = np.linalg.norm(self.x[3:])
            dt_scale = min(self.dt / 60.0, 10.0)  # Limit scaling for stability
            process_scale = 1e-6 * vel_magnitude * dt_scale
            dx_process = np.random.normal(0, process_scale, size=self.dim_x)

        # Adaptive process noise with proper scaling
        Q_innovation = np.outer(dx_process, dx_process)
        self.Q_adaptive = rho * self.Q_adaptive + (1 - rho) * Q_innovation

        # CRITICAL: Apply realistic bounds to prevent divergence
        self._apply_adaptive_bounds()

    def _apply_adaptive_bounds(self):
        """Apply realistic bounds to adaptive noise matrices - DIMENSION ADAPTIVE"""

        # For satellite systems (6D), use GPS-based bounds
        if self.dim_x == 6 and self.dim_z == 6:
            # Measurement noise bounds (based on GPS/GNSS capabilities)
            R_min = np.diag([10.0**2] * 3 + [0.05**2] * 3)  # Min: 10m pos, 0.05m/s vel
            R_max = np.diag([200.0**2] * 3 + [2.0**2] * 3)  # Max: 200m pos, 2m/s vel

            # Process noise bounds (based on orbital mechanics)
            altitude_est = np.linalg.norm(self.x[:3]) / 1000  # km
            if altitude_est < 600:  # LEO
                Q_scale = 1e-4
            elif altitude_est < 1500:  # MEO
                Q_scale = 1e-5
            else:  # GEO
                Q_scale = 1e-6

            Q_max = np.eye(self.dim_x) * Q_scale
            Q_min = np.eye(self.dim_x) * (Q_scale * 1e-3)

        else:
            # For general systems, use conservative bounds based on current values
            R_diag = np.diag(self.R_adaptive)
            Q_diag = np.diag(self.Q_adaptive)

            # Measurement noise bounds (10x smaller to 100x larger than current)
            R_min = np.diag(np.maximum(R_diag * 0.1, 1e-6))
            R_max = np.diag(R_diag * 100.0)

            # Process noise bounds (1000x smaller to 100x larger than current)
            Q_min = np.diag(np.maximum(Q_diag * 0.001, 1e-12))
            Q_max = np.diag(Q_diag * 100.0)

        # Apply bounds with proper size matching
        if self.R_adaptive.shape == R_min.shape:
            self.R_adaptive = np.maximum(np.minimum(self.R_adaptive, R_max), R_min)

        if self.Q_adaptive.shape == Q_min.shape:
            self.Q_adaptive = np.maximum(np.minimum(self.Q_adaptive, Q_max), Q_min)

    def _innovation_based(self) -> None:
        w = self.params.innovation_window
        if len(self.innovation_history) < w:
            return
        V = np.array(self.innovation_history[-w:]).T
        C = V @ V.T / w
        self.R_adaptive = 0.9 * self.R_adaptive + 0.1 * C
        self._ensure_pd()

    def _variational_bayes(self) -> None:
        self.R_adaptive *= 1.01
        self._ensure_pd()

    def _ensure_pd(self) -> None:
        self.Q_adaptive += np.eye(self.dim_x) * 1e-12
        self.R_adaptive += np.eye(self.dim_z) * 1e-12

    def get_nis_statistics(self) -> dict:
        if not self.nis_history:
            return {}
        arr = np.asarray(self.nis_history)
        mean = float(arr.mean())
        ci_lower, ci_upper = np.percentile(arr, [2.5, 97.5])
        dof = self.dim_z
        chi2_passed = 0.025 < chi2.cdf(mean, dof) < 0.975
        return {
            "mean": mean,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "chi2_passed": bool(chi2_passed),
        }

    def reset_adaptive_parameters(self) -> None:
        self.Q_adaptive = self.Q.copy()
        self.R_adaptive = self.R.copy()
        self.innovation_history.clear()
        self.nis_history.clear()
        self.condition_numbers.clear()
        self.x_prior = None

    def set_state(self, x: np.ndarray, P: np.ndarray):
        """Set filter state and covariance"""
        self.x = x.copy()
        self.P = P.copy()

    def set_noise_matrices(self, Q: np.ndarray, R: np.ndarray):
        """Set process and measurement noise covariances"""
        self.Q = Q.copy()
        self.R = R.copy()
        self.Q_adaptive = Q.copy()
        self.R_adaptive = R.copy()
