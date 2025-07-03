# aukf.py  ─── minimal but working UKF helper  ─────────────────────────────

from __future__ import annotations
import orekit, numpy as np

# ── Start the JVM (safe to call twice) ────────────────────────────────
try:                        # on first import it starts the JVM
    orekit.initVM()
except RuntimeError:        # second import → “already started”
    pass

# ── Orekit bits needed for sigma-point propagation ───────────────────────
from org.orekit.utils   import PVCoordinates, Constants
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.orbits  import CartesianOrbit
from org.orekit.frames  import FramesFactory
from org.orekit.time    import AbsoluteDate

# ── Unscented transform helper (unchanged) ────────────────────────────────
def _sigma_points(x: np.ndarray,
                  P: np.ndarray,
                  alpha: float, beta: float, kappa: float):
    n   = len(x)
    lam = alpha**2 * (n + kappa) - n
    U   = np.linalg.cholesky((n + lam) * P)

    chi = [x] + [x + U[i] for i in range(n)] + [x - U[i] for i in range(n)]
    chi = np.vstack(chi)

    W_m = np.full(2*n+1, 1/(2*(n+lam)))
    W_c = W_m.copy()
    W_m[0] = lam/(n+lam)
    W_c[0] = W_m[0] + (1 - alpha**2 + beta)
    return chi, W_m, W_c


# ── UKF class  ────────────────────────────────────────────────────────────
class UnscentedKalman:
    """
    Very thin UKF wrapper around an Orekit propagator supplied at init.
    Prediction: every sigma-point is advanced with that propagator.
    Update: basic linear measurement model z = x (because we feed
    position+velocity directly).
    """

    # ---------------------------------------------------------------------
    def __init__(self,
                 propagator,
                 meas_cols: list[str],
                 *,
                 dim_x: int = 6,
                 dim_z: int = 6,
                 alpha: float = 1e-3,
                 beta : float = 2.0,
                 kappa: float = 0.0,
                 q0:   float = 1e-2,
                 r0:   float = 25.0,
                 adaptive: str | None = None):

        # keep dimensions handy  🔧
        self.dim_x = dim_x
        self.dim_z = dim_z

        # store settings
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.adaptive = adaptive
        self.cols     = meas_cols
        self.prop     = propagator

        # state holders
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x) * q0
        self.R = np.eye(dim_z) * r0

        # book-keeping
        self.history = []          # (AbsoluteDate, x̂) tuples
        self._current_date = None  # updated in `step`

    # ---------------------------------------------------------------------
    def init_from_measurement(self, t0_sec: float, z0: np.ndarray):
        """Initialize filter at epoch‐0 using first measurement."""
        self.x = z0.copy()
        self.P = np.eye(self.dim_x) * 1e4
        self._current_date = AbsoluteDate.J2000_EPOCH.shiftedBy(float(t0_sec))
        self.history = [(self._current_date, self.x.copy())]

    # ---------------------------------------------------------------------
    def _propagate_sigma(self, sig: np.ndarray, dt: float) -> np.ndarray:
        """Advance one sigma-point `dt` seconds with Orekit."""
        # Build Orekit orbit from Cartesian (EME2000)
        frame  = FramesFactory.getEME2000()
        vec3   = lambda a: Vector3D(float(a[0]), float(a[1]), float(a[2]))
        pv     = PVCoordinates(vec3(sig[:3]), vec3(sig[3:]))        
        orbit0 = CartesianOrbit(pv, frame, self._current_date, Constants.WGS84_EARTH_MU)

        tspan, states = self.prop.propagate_from_state(orbit0, dt, step=dt)
        pv1   = states[-1].getPVCoordinates()

        xyz = lambda v: np.array((v.getX(), v.getY(), v.getZ()), dtype=float)
        return np.hstack([xyz(pv1.getPosition()), xyz(pv1.getVelocity())])
        
    # ---------------------------------------------------------------------
    def step(self, dt: float, Z: np.ndarray):
        """Predict `dt` s ahead then update with measurement `Z`."""
        # σ-points
        chi, Wm, Wc = _sigma_points(self.x, self.P,
                                    self.alpha, self.beta, self.kappa)

        # --- PREDICT -------------------------------------------------------
        chi_pred = np.vstack([self._propagate_sigma(s, dt) for s in chi])
        x_pred   = chi_pred.T @ Wm
        P_pred   = ((chi_pred - x_pred).T * Wc) @ (chi_pred - x_pred) + self.Q

        # --- UPDATE (identity H) ------------------------------------------
        y   = Z - x_pred
        Pzz = P_pred + self.R
        K   = Pxz @ np.linalg.inv(Pzz)

        self.x = x_pred + K @ y
        self.P = P_pred - K @ Pzz @ K.T

        if self.adaptive == "iae":
            self._iae_update_covariances(y, Pzz)

    # --------------------------------------------------------------
    def _iae_update_covariances(self, innov: np.ndarray, S: np.ndarray):
        """
        Innovation-Based Adaptive Estimation (Li & Zhao 2014).

        Parameters
        ----------
        innov : z_k − H x̂_k|k−1      (here H = I)
        S     : innovation covariance (Pzz)
        """
        # ---- adapt R -----------------------------------------------------
        alpha = 0.05                    # [0.01 … 0.1]  ← feel free to tune
        self.R = (1 - alpha) * self.R + alpha * np.outer(innov, innov)

        # ---- adapt Q -----------------------------------------------------
        phi   = 0.995                   # fading-memory  (0.98 … 1.0)
        self.Q /= phi                   # inflate   (φ < 1) ⇒ larger Q

