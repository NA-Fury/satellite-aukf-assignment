# aukf.py  ─── minimal but working UKF helper with robust jitter ────────────

from __future__ import annotations
import numpy as np
import json

try: orekit.initVM() 
except RuntimeError: pass

from org.orekit.utils   import PVCoordinates, Constants
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.orbits  import CartesianOrbit
from org.orekit.frames  import FramesFactory
from org.orekit.time    import AbsoluteDate

class UnscentedKalman:
    """
    Constant-velocity UKF (CV-UKF).
    Innovation-Based Adaptive Estimation per Li & Zhao (2014, doi:10.1109/TITS.2014.2303118).
    """

    def __init__(self,
                 propagator,
                 meas_cols: list[str],
                 *,
                 alpha: float = 1e-3,
                 beta : float = 2.0,
                 kappa: float = 0.0,
                 q0:   float = 1e-2,
                 r0:   float = 25.0,
                 adaptive: str | None = None):
        self.prop, self.cols = propagator, meas_cols
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.adaptive = adaptive

        self.dim_x, self.dim_z = 6, len(meas_cols)
        self.x  = np.zeros(self.dim_x)
        self.P  = np.eye(self.dim_x)*1e4
        self.Q  = np.eye(self.dim_x)*q0
        self.R  = np.eye(self.dim_z)*r0

        self.history    = []     # (date, x) tuples
        self._last_innov = None  # for external logging

    def init_from_measurement(self, t0_sec: float, z0: np.ndarray):
        self.x = z0.copy()
        self.P = np.eye(self.dim_x)*1e4
        self._current_date = AbsoluteDate.J2000_EPOCH.shiftedBy(t0_sec)
        self.history = [(self._current_date, self.x.copy())]

    def _sigma_points(self, x, P):
        n, α, β, κ = self.dim_x, self.alpha, self.beta, self.kappa
        λ = α**2*(n+κ) - n

        P = 0.5*(P+P.T) + 1e-9*np.eye(n)
        base = n + λ
        eps = 1e-9 * np.trace(P)/n
        for _ in range(6):
            try:
                L = np.linalg.cholesky(base*(P+eps*np.eye(n)))
                break
            except:
                eps *= 10
        else:
            L = np.linalg.cholesky(np.eye(n))

        chi = np.vstack([x] + [x + L[i] for i in range(n)] + [x - L[i] for i in range(n)])
        Wm = np.full(2*n+1, 1/(2*base))
        Wc = Wm.copy()
        Wm[0] = λ/base
        Wc[0] = Wm[0] + (1-α**2+β)
        return chi, Wm, Wc

    def _predict_cv(self, σ, dt):
        p = σ[:3] + σ[3:]*dt
        return np.hstack([p, σ[3:]])

    def step(self, dt: float, Z: np.ndarray):
        χ, Wm, Wc = self._sigma_points(self.x, self.P)
        χp = np.vstack([self._predict_cv(s, dt) for s in χ])
        x_p = χp.T @ Wm
        P_p = ((χp - x_p).T * Wc) @ (χp - x_p) + self.Q

        y    = Z - x_p
        self._last_innov = y         # expose innovation
        Pzz  = P_p + self.R
        Pxz  = P_p
        K    = Pxz @ np.linalg.inv(Pzz)

        I     = np.eye(self.dim_x)
        self.x = x_p + K@y
        self.P = (I-K)@P_p@(I-K).T + K@self.R@K.T
        self.P = 0.5*(self.P+self.P.T) + 1e-6*np.eye(self.dim_x)

        if self.adaptive == "iae":
            α=0.05
            self.R = (1-α)*self.R + α*np.outer(y,y)

        # record
        self._current_date = self._current_date.shiftedBy(dt)
        self.history.append((self._current_date, self.x.copy()))
