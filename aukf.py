# aukf.py  ─── minimal but working UKF helper with robust jitter ────────────

from __future__ import annotations
import numpy as np

def _sigma_points(x: np.ndarray,
                  P: np.ndarray,
                  alpha: float,
                  beta: float,
                  kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sigma points and weights for the Unscented Transform.
    """
    n = x.shape[0]
    lam = alpha**2 * (n + kappa) - n

    # ensure symmetry and add tiny jitter for numeric stability
    P = 0.5 * (P + P.T) + 1e-9 * np.eye(n)
    base = n + lam

    # try Cholesky with increasing diagonal jitter
    eps = 1e-9 * np.trace(P) / n
    for _ in range(6):
        try:
            L = np.linalg.cholesky(base * (P + eps * np.eye(n)))
            break
        except np.linalg.LinAlgError:
            eps *= 10
    else:
        L = np.linalg.cholesky(np.eye(n))

    # build sigma points
    chi = np.vstack(
        [x] +
        [x + L[i] for i in range(n)] +
        [x - L[i] for i in range(n)]
    )

    # weights
    Wm = np.full(2 * n + 1, 1.0 / (2 * base))
    Wc = Wm.copy()
    Wm[0] = lam / base
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    return chi, Wm, Wc


class UnscentedKalman:
    """
    Constant-velocity UKF (CV-UKF) with optional Innovation-based Adaptive Estimation (IAE).
    """

    def __init__(self,
                 meas_cols: list[str],
                 *,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0,
                 q0: float = 1e-2,
                 r0: float = 25.0,
                 adaptive: str | None = None):
        self.cols = meas_cols
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.adaptive = adaptive

        # state dimension = 6 (px, py, pz, vx, vy, vz)
        self.dim_x = 6
        # measurement dimension matches number of measurement columns
        self.dim_z = len(meas_cols)

        # initial state & covariances
        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 1e4
        self.Q = np.eye(self.dim_x) * q0
        self.R = np.eye(self.dim_z) * r0

        # storage
        self.history: list[tuple[float, np.ndarray]] = []
        self._last_innov: np.ndarray | None = None

    def init_from_measurement(self, t0_sec: float, z0: np.ndarray) -> None:
        """
        Initialize the filter state from the first measurement.
        """
        self.x = z0.copy()
        self.P = np.eye(self.dim_x) * 1e4
        # record time and state
        self.history = [(t0_sec, self.x.copy())]

    def step(self, dt: float, Z: np.ndarray) -> None:
        """
        Perform one UKF predict-update cycle with constant-velocity model.
        """
        # 1) generate sigma points
        chi, Wm, Wc = _sigma_points(self.x, self.P,
                                     self.alpha, self.beta, self.kappa)

        # 2) propagate each sigma point under constant velocity
        chi_p = np.vstack([
            np.hstack([pt[:3] + pt[3:] * dt, pt[3:]])
            for pt in chi
        ])

        # 3) recombine to get predicted mean & covariance
        x_p = chi_p.T @ Wm
        P_p = ((chi_p - x_p).T * Wc) @ (chi_p - x_p) + self.Q

        # 4) innovation
        y = Z - x_p
        self._last_innov = y
        Pzz = P_p + self.R
        Pxz = P_p

        # 5) Kalman gain & update
        K = Pxz @ np.linalg.inv(Pzz)
        I = np.eye(self.dim_x)
        self.x = x_p + K @ y
        self.P = (I - K) @ P_p @ (I - K).T + K @ self.R @ K.T
        # enforce symmetry & jitter
        self.P = 0.5 * (self.P + self.P.T) + 1e-6 * np.eye(self.dim_x)

        # 6) optional adaptive R update
        if self.adaptive == "iae":
            alpha_iae = 0.05
            self.R = (1 - alpha_iae) * self.R + alpha_iae * np.outer(y, y)

        # 7) record time-step and state estimate
        last_t = self.history[-1][0] if self.history else 0.0
        self.history.append((last_t + dt, self.x.copy()))
