"""
Setup script for Orekit environment
===================================
This script helps set up the Orekit data directory and verify the installation.
"""

import os
import sys

def setup_orekit_data():
    """Set up Orekit data directory."""
    # First try to import orekit
    try:
        import orekit
        print("✓ Orekit is installed")
    except ImportError:
        print("✗ Orekit is not installed!")
        print("Please install it with: pip install orekit")
        return False
    
    # Initialize the JVM
    try:
        orekit.initVM()
        print("✓ JVM initialized")
    except Exception as e:
        print(f"✗ Failed to initialize JVM: {e}")
        return False
    
    # Check for orekit-data directory
    data_dir = os.path.abspath("orekit-data")
    if os.path.exists(data_dir):
        print(f"✓ Found orekit-data directory at: {data_dir}")
    else:
        print(f"✗ orekit-data directory not found at: {data_dir}")
        print("  Run download_orekit_data.py first to download the data")
        return False
    
    # Set up the data context
    try:
        from org.orekit.data import DataProvidersManager, DirectoryCrawler
        from java.io import File
        
        manager = DataProvidersManager.getInstance()
        crawler = DirectoryCrawler(File(data_dir))
        manager.clearProviders()
        manager.addProvider(crawler)
        
        print("✓ Orekit data providers configured")
        return True
        
    except Exception as e:
        print(f"✗ Failed to configure Orekit data providers: {e}")
        return False

def test_orekit_propagator():
    """Test that Orekit propagator works."""
    try:
        from org.orekit.frames import FramesFactory
        from org.orekit.utils import Constants
        from org.orekit.orbits import KeplerianOrbit, PositionAngle
        from org.orekit.propagation.analytical import KeplerianPropagator
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        import math
        
        # Get reference frames
        gcrf = FramesFactory.getGCRF()
        utc = TimeScalesFactory.getUTC()
        
        # Create a test orbit (similar to ISS)
        a = 6800.0e3  # semi-major axis in meters
        e = 0.001     # eccentricity
        i = math.radians(51.6)  # inclination
        omega = math.radians(0.0)   # perigee argument
        raan = math.radians(0.0)    # RAAN
        lv = math.radians(0.0)      # true anomaly
        
        # Initial date
        initialDate = AbsoluteDate(2024, 1, 1, 0, 0, 0.0, utc)
        
        # Create orbit
        initialOrbit = KeplerianOrbit(a, e, i, omega, raan, lv,
                                      PositionAngle.TRUE, gcrf, initialDate,
                                      Constants.WGS84_EARTH_MU)
        
        # Create propagator
        propagator = KeplerianPropagator(initialOrbit)
        
        # Propagate for 60 seconds
        finalState = propagator.propagate(initialDate.shiftedBy(60.0))
        
        print("✓ Orekit propagator test successful!")
        print(f"  Initial position: {initialOrbit.getPVCoordinates().getPosition()}")
        print(f"  Final position: {finalState.getPVCoordinates().getPosition()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Orekit propagator test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("Orekit Environment Setup")
    print("=" * 50)
    
    # Set up Orekit data
    if not setup_orekit_data():
        print("\nSetup failed! Please address the issues above.")
        return
    
    print("\nTesting Orekit propagator...")
    if test_orekit_propagator():
        print("\n✓ Orekit is properly configured and working!")
    else:
        print("\n✗ Orekit test failed. Check your installation.")

if __name__ == "__main__":
    main()