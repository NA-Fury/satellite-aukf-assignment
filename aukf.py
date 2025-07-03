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
from org.orekit.utils import PVCoordinates, Vector3D, Constants
from org.orekit.orbits import CartesianOrbit
from org.orekit.frames import FramesFactory

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
    
    # keep track of the current absolute date
    def init_from_measurement(self, t0_sec: float, z0: np.ndarray):
        self.x = z0.copy()
        self.P = np.eye(self.dim_x) * 1e4
        self._current_date = FramesFactory.getEME2000()\
                            .getEpoch()\
                            .shiftedBy(float(t0_sec))
        self.history = [(self._current_date, self.x.copy())]

    # ------------------------------------------------------------------
    # 2.2  Predict-then-update for one epoch
    # ------------------------------------------------------------------
    def step(self, dt: float, Z: np.ndarray):
        """
        UKF predict-update for a single epoch.
        """
        # ----- SIGMA-POINTS --------------------------------------------------
        χ, Wm, Wc = _sigma_points(self.x, self.P,
                                  self.alpha, self.beta, self.kappa)

        # ----- PREDICT ------------------------------------------------------
        χ_pred = np.vstack([self._propagate_sigma(sig, dt) for sig in χ])
        x_pred = χ_pred.T @ Wm
        P_pred = (χ_pred - x_pred).T @ np.diag(Wc) @ (χ_pred - x_pred) + self.Q

        # ----- UPDATE -------------------------------------------------------
        y  = Z - x_pred                 # residual
        S  = P_pred + self.R
        K  = P_pred @ np.linalg.inv(S)  # Kalman gain
        self.x = x_pred + K @ y
        self.P = (np.eye(self.dim_x) - K) @ P_pred

        # advance absolute date for next call
        self._current_date = self._current_date.shiftedBy(float(dt))
        self.history.append((self._current_date, self.x.copy()))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _propagate_sigma(self, sig: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate one sigma-point `sig` forward by `dt` [s] using Orekit.
        Returns the 6-element Cartesian state in EME2000.
        """
        # build Orekit PV from the sigma-point
        pv0 = PVCoordinates(Vector3D(*sig[:3]), Vector3D(*sig[3:]))

        # CartesianOrbit in EME2000
        eme2000 = FramesFactory.getEME2000()
        orbit0  = CartesianOrbit(pv0, eme2000,
                                 self._current_date,
                                 Constants.WGS84_EARTH_MU)

        # 1-step propagation
        tspan, states = self.prop.propagate_from_state(orbit0,
                                                       duration=float(dt),
                                                       step=float(dt))
        pv1 = states[-1].getPVCoordinates()

        # return as numpy vector
        return np.hstack([pv1.getPosition().toArray(),
                          pv1.getVelocity().toArray()])

    def _sage_husa(self, y, K, S):
        """Very light Sage–Husa adaptive noise estimator."""
        gamma = 0.01
        self.Q += gamma * (np.outer(self.x, self.x) - self.Q)
        self.R += gamma * (np.outer(y, y) - S - self.R)

    def propagate_sigma_points(self, chi, dt_sec, propagator):
        """
        Propagate every sigma point forward by dt_sec with Orekit.
    
        Parameters
        ----------
        chi : ndarray shape (2n+1, n)
            Sigma-point matrix at time k.
        dt_sec : float
            Δt to propagate [s].
        propagator : OrbitPropagator
            Wrapper you wrote earlier.
    
        Returns
        -------
        chi_fwd : ndarray shape (2n+1, n)
            Propagated sigma-points at k+1.
        """
        out = np.empty_like(chi)
        for i, sp in enumerate(chi):
            # build CartesianOrbit from the 6-element vector
            pos = Vector3D(sp[0], sp[1], sp[2])
            vel = Vector3D(sp[3], sp[4], sp[5])
            pv  = PVCoordinates(pos, vel)
            orbit = CartesianOrbit(
                pv,
                FramesFactory.getEME2000(),
                propagator.propagator.getInitialState().getDate(),  # epoch of χ
                Constants.WGS84_EARTH_MU,
            )
            tspan, states = propagator.propagate_from_state(orbit, dt_sec, step=dt_sec)
            pv_new = states[-1].getPVCoordinates()
            out[i] = np.r_[
                pv_new.getPosition().toArray(),
                pv_new.getVelocity().toArray(),
            ]
        return out

    def predict(self, dt, propagator):
        """Unscented transform through the dynamics f(x)."""
        chi, Wm, Wc = _sigma_points(self.x, self.P, self.alpha, self.kappa, self.beta)
        chi_fwd      = self.propagate_sigma_points(chi, dt, propagator)
    
        # predicted mean
        self.x = np.sum(Wm[:, None] * chi_fwd, axis=0)
    
        # predicted covariance
        diff   = chi_fwd - self.x
        self.P = diff.T @ (Wc[:, None] * diff) + self.Q

    def update(self, z):
        """
        Standard UKF measurement update with full-state observation.
    
        z : array-like shape (6,)
            GPS position + velocity in metres / m·s⁻¹
        """
        chi, Wm, Wc = _sigma_points(self.x, self.P, self.alpha, self.kappa, self.beta)
        # measurement model is identity ⇒ Z = chi
        Z_sigma = chi.copy()
    
        z_pred = np.sum(Wm[:, None] * Z_sigma, axis=0)
        diff_z = Z_sigma - z_pred
        Pzz    = diff_z.T @ (Wc[:, None] * diff_z) + self.R
        Pxz    = (chi - self.x).T @ (Wc[:, None] * diff_z)
    
        K = Pxz @ np.linalg.inv(Pzz)          # Kalman gain
        self.x += K @ (z - z_pred)
        self.P -= K @ Pzz @ K.T
    
        # (optional) adaptive_covariance update
        if self.adaptive:
            self.update_QR(z, z_pred)



