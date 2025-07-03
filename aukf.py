"""
aukf.py   — Adaptive Unscented Kalman Filter for GNSS + Orekit

Public API
----------
UKF = UnscentedKalman(
        propagator,           # OrbitPropagator instance (already configured)
        meas_cols,            # list of 6 str: position_x … velocity_z
        q0=1e-4, r0=10.0,     # initial process / measurement noise scalars
        alpha=1e-3, beta=2.0, kappa=0.0,
        adaptive="sage-husa", # or "iae" or None
    )
UKF.step(dt, Z)  # predict-then-update for one epoch
UKF.history      # list of (t, state_vector, cov) tuples
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------
# 1.  Σ-point generator  (private helper)
# --------------------------------------------------------------------------
def _sigma_points(x: np.ndarray, P: np.ndarray,
                  alpha: float, beta: float, kappa: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Julier–Uhlmann symmetric sigma set.

    Returns
    -------
    chi  : (2n+1, n)  sigma vectors
    Wm   : (2n+1,)    weights for state / mean
    Wc   : (2n+1,)    weights for covariance
    """
    n = x.size
    lam = alpha**2 * (n + kappa) - n
    U = np.linalg.cholesky((n + lam) * P)

    chi = np.vstack([x,
                     x[None] +  U,
                     x[None] -  U])

    Wm = np.full(2*n + 1, 1. / (2*(n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    return chi, Wm, Wc


# --------------------------------------------------------------------------
# 2.  Unscented Kalman class
# --------------------------------------------------------------------------
class UnscentedKalman:
    def __init__(self,
                 propagator,          # OrbitPropagator
                 meas_cols: list[str],
                 q0: float = 1e-4,
                 r0: float = 10.0,
                 alpha: float = 1e-3,
                 beta : float = 2.0,
                 kappa: float = 0.0,
                 adaptive: str | None = "sage-husa",
                 ):
        self.prop      = propagator
        self.cols      = meas_cols
        self.n         = 6                    # state dim  (x, y, z, vx, vy, vz)
        self.m         = 6                    # measurement dim  (same)
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

        self.x   = None                       # state vector
        self.P   = None                       # covariance
        self.Q   = np.eye(self.n) * q0
        self.R   = np.eye(self.m) * r0
        self.history = []                     # [(t, x, P), …]
        self.adaptive = adaptive

    # ------------------------------------------------------------------
    # 2.1  initialise from first GNSS epoch
    # ------------------------------------------------------------------
    def init_from_measurement(self, t0, Z0):
        self.x = Z0.copy()                    # naïve state = measurement
        self.P = np.eye(self.n) * 1e2         # 100 m² position, (1 m/s)² vel
        self.history.append((t0, self.x.copy(), self.P.copy()))

    # ------------------------------------------------------------------
    # 2.2  Predict-then-update for one epoch
    # ------------------------------------------------------------------
    def step(self, dt: float, Z: np.ndarray):
        χ, Wm, Wc = _sigma_points(self.x, self.P,
                                  self.alpha, self.beta, self.kappa)

        # ---- PREDICT: propagate each σ-point with Orekit --------------
        χ_pred = np.vstack([self._propagate_sigma(sig, dt) for sig in χ])
        x_pred = χ_pred.T @ Wm
        P_pred = (χ_pred - x_pred).T @ np.diag(Wc) @ (χ_pred - x_pred) + self.Q

        # ---- UPDATE: transform to measurement space (identity here) ---
        Z_pred = χ_pred                  # same coords
        z_pred = Z_pred.T @ Wm
        S      = (Z_pred - z_pred).T @ np.diag(Wc) @ (Z_pred - z_pred) + self.R
        Cxz    = (χ_pred - x_pred).T  @ np.diag(Wc) @ (Z_pred - z_pred)
        K      = Cxz @ np.linalg.inv(S)

        y      = Z - z_pred               # innovation
        self.x = x_pred + K @ y
        self.P = P_pred - K @ S @ K.T

        # ---- optional adaptive Q/R  -----------------------------------
        if self.adaptive == "sage-husa":
            self._sage_husa(y, K, S)

        self.history.append((self.history[-1][0] + dt,  # time
                             self.x.copy(), self.P.copy()))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _propagate_sigma(self, sig: np.ndarray, dt: float) -> np.ndarray:
        """Wrap OrbitPropagator.propagate_from_state for a σ-point."""
        from org.orekit.orbits import CartesianOrbit
        from org.orekit.frames import FramesFactory
        from org.orekit.utils import Constants
        from org.orekit.time   import AbsoluteDate
        # build Orekit orbit from sig (EME2000 frame)
        eme2000 = FramesFactory.getEME2000()
        pv = self.prop.Vector3D(*sig[:3]), self.prop.Vector3D(*sig[3:])
        date0 = self.prop.AbsoluteDate(
                *self.history[-1][:1],  # current AbsoluteDate
            )
        orbit = CartesianOrbit(self.prop.PVCoordinates(*pv),
                               eme2000, date0, Constants.WGS84_EARTH_MU)
        tspan, states = self.prop.propagate_from_state(orbit, dt, step=dt)
        final = states[-1].getPVCoordinates()
        return np.array([
            final.getPosition().getX(),
            final.getPosition().getY(),
            final.getPosition().getZ(),
            final.getVelocity().getX(),
            final.getVelocity().getY(),
            final.getVelocity().getZ(),
        ], dtype=float)

    def _sage_husa(self, y, K, S):
        """Very light Sage–Husa adaptive noise estimator."""
        gamma = 0.01
        self.Q += gamma * (np.outer(self.x, self.x) - self.Q)
        self.R += gamma * (np.outer(y, y) - S - self.R)
