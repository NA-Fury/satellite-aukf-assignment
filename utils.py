# utils.py  — only data I/O + plotting

import os, pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import orekit                     # ① import orekit
orekit.initVM()                   # ② start the JVM (once per process)

from java.io import File          # ③ now safe – ‘java’ package exists
from orekit import JArray_double
from orekit.pyhelpers import setup_orekit_curdir

from org.orekit.data import DataContext, DirectoryCrawler
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.bodies         import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.drag     import DragForce, IsotropicDrag
from org.orekit.forces.gravity  import HolmesFeatherstoneAttractionModel, SolidTides
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames         import FramesFactory
from org.orekit.models.earth.atmosphere      import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.time           import TimeScalesFactory
from org.orekit.utils          import Constants, IERSConventions

try:
    setup_orekit_curdir()
except RuntimeError:
    pass

DATA_DIR = pathlib.Path(os.getenv("OREKIT_DATA_PATH", "orekit-data")).resolve()
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Orekit data dir {DATA_DIR} missing.\n"
                            "Clone https://gitlab.orekit.org/orekit/orekit-data (with git-lfs) "
                            "or run orekit.pyhelpers.download_orekit_data_curdir()")

dpm = DataContext.getDefault().getDataProvidersManager()
dpm.clearProviders()
dpm.addProvider(DirectoryCrawler(File(str(DATA_DIR))))
print("✔️  Orekit data registered:", DATA_DIR)

class OrbitPropagator:
    """
    Orbit propagator class that utilizes Orekit's NumericalPropagator.
    """
    def __init__(self,
                 satellite_mass: float = 260.0,
                 cross_section: float = 3.2 * 1.6,
                 drag_coeff: float = 2.2,
                 degree: int = 2,
                 torder: int = 2,
                 position_tolerance: float = 1e-2):
        self.mass              = satellite_mass
        self.cross_section     = cross_section
        self.drag_coeff        = drag_coeff
        self.degree            = degree
        self.torder            = torder
        self.position_tolerance= position_tolerance

        self.sun   = CelestialBodyFactory.getSun()
        self.moon  = CelestialBodyFactory.getMoon()
        itrf       = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        r_e        = Constants.IERS2010_EARTH_EQUATORIAL_RADIUS
        self.earth= OneAxisEllipsoid(r_e, Constants.IERS2010_EARTH_FLATTENING, itrf)
        self.sat_model = IsotropicDrag(self.cross_section, self.drag_coeff)
        self.sw_data   = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
        self._propagator = None

    def setup(self, initial_orbit, step_hint: float = 1.0):
        tol = NumericalPropagator.tolerances(
            self.position_tolerance,
            initial_orbit,
            initial_orbit.getType()
        )
    
        min_step = max(5e-7, 1e-4 * step_hint)
        max_step = max(180.0, 180.0 * step_hint)
    
        abs_tol = JArray_double.cast_(tol[0])
        rel_tol = JArray_double.cast_(tol[1])
    
        integ = DormandPrince853Integrator(
            float(min_step),
            float(max_step),
            abs_tol,
            rel_tol
        )
        integ.setInitialStepSize(float(max(min_step*20, 1e-4)))

        self._propagator = NumericalPropagator(integ)
        self._propagator.setOrbitType(initial_orbit.getType())

        grav = GravityFieldFactory.getConstantNormalizedProvider(
            self.degree, self.torder, initial_orbit.getDate())
        self._propagator.addForceModel(
            HolmesFeatherstoneAttractionModel(self.earth.getBodyFrame(), grav))
        self._propagator.addForceModel(
            SolidTides(self.earth.getBodyFrame(),
                       grav.getAe(), grav.getMu(), grav.getTideSystem(),
                       IERSConventions.IERS_2010,
                       TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True),
                       [self.sun, self.moon]))
        self._propagator.addForceModel(
            DragForce(NRLMSISE00(self.sw_data, self.sun, self.earth),
                      self.sat_model))

    def propagate(self, initial_orbit, duration: float, step: float = None):
        if self._propagator is None:
            self.setup(initial_orbit, step if step else duration)

        state0 = SpacecraftState(initial_orbit, self.mass)
        self._propagator.setInitialState(state0)

        if step is None:
            step = duration
        ts = np.linspace(0, duration, int(duration / step) + 1)
        dates = [initial_orbit.getDate().shiftedBy(float(dt)) for dt in ts]
        states= [self._propagator.propagate(d) for d in dates]
        return dates, states

    def propagate_from_state(self, initial_orbit, duration: float, step: float = None):
        """Alias for propagate() kept for older notebooks."""
        return self.propagate(initial_orbit, duration, step)


# Convenience I/O & plotting
def load_clean(path="GPS_clean.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)

def plot_ground_track(xs: np.ndarray, title="Ground Track (km)"):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(xs[:,0]/1e3, xs[:,1]/1e3, lw=0.6)
    ax.set_aspect('equal')
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]')
    ax.set_title(title)
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
def _fake_init_orbit(self):
    """
    Very rough Cartesian orbit built from the first clean GPS row.
    Only to seed truth-propagation; UKF still converges.
    """
    df = pd.read_parquet(
        "GPS_clean.parquet",
        columns=[
            "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z",
        ],
    )

    # ---- CAST to Python float (C double) ---------------------------
    raw = df.iloc[0].values
    row = [float(v) for v in raw]
    
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.utils   import PVCoordinates, Constants
    from org.orekit.orbits  import CartesianOrbit
    from org.orekit.frames  import FramesFactory
    from org.orekit.time    import AbsoluteDate

    pv = PVCoordinates(
        Vector3D(*row[:3]),
        Vector3D(*row[3:]),
    )
    return CartesianOrbit(
        pv,
        FramesFactory.getEME2000(),
        AbsoluteDate.J2000_EPOCH,
        Constants.WGS84_EARTH_MU,
    )

# bind  helper onto the class so old notebooks still work
OrbitPropagator._fake_init_orbit = _fake_init_orbit