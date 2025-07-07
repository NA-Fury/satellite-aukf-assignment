"""
Analyze GPS data to understand units and fix conversion issues
=============================================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import validate_orbit_parameters, R_EARTH

def analyze_gps_data():
    """Analyze GPS data to understand the units and typical values."""
    
    # Load data
    try:
        df = pd.read_parquet("GPS_clean.parquet")
        cols = json.load(open("meas_cols.json"))
        print(f"Loaded {len(df):,} measurements")
        print(f"Columns: {cols}")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")
        print(f"Duration: {df['time'].max() - df['time'].min()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract position and velocity data
    pos_cols = cols[:3]
    vel_cols = cols[3:]
    
    pos_data = df[pos_cols].values
    vel_data = df[vel_cols].values
    
    print("\n=== Raw Data Statistics ===")
    print(f"Position stats (assumed km):")
    print(f"  Mean: {np.mean(pos_data, axis=0)}")
    print(f"  Std:  {np.std(pos_data, axis=0)}")
    print(f"  Min:  {np.min(pos_data, axis=0)}")
    print(f"  Max:  {np.max(pos_data, axis=0)}")
    
    print(f"\nVelocity stats (assumed dm/s):")
    print(f"  Mean: {np.mean(vel_data, axis=0)}")
    print(f"  Std:  {np.std(vel_data, axis=0)}")
    print(f"  Min:  {np.min(vel_data, axis=0)}")
    print(f"  Max:  {np.max(vel_data, axis=0)}")
    
    # Calculate orbital radius
    radii = np.linalg.norm(pos_data, axis=1)
    print(f"\nOrbital radius (if km):")
    print(f"  Mean: {np.mean(radii):.1f} km")
    print(f"  Std:  {np.std(radii):.1f} km")
    print(f"  Min:  {np.min(radii):.1f} km")
    print(f"  Max:  {np.max(radii):.1f} km")
    
    # Check if these could be Earth satellites
    if np.mean(radii) > 1e6:  # If mean > 1 million km
        print("\n⚠️  WARNING: Position values are too large for Earth satellites!")
        print("   Checking if positions might be in meters instead of km...")
        
        # Try interpreting as meters
        radii_m = radii * 1000  # Convert to meters if they were km
        altitudes_km = (radii_m - R_EARTH) / 1000
        print(f"\nIf positions are in meters (not km):")
        print(f"  Mean altitude: {np.mean(altitudes_km):.1f} km")
        print(f"  This would be underground! Data units are likely wrong.")
        
        # Maybe the data is already in meters?
        radii_direct_m = radii  # Treat the values as meters
        altitudes_direct_km = (radii_direct_m - R_EARTH) / 1000
        print(f"\nIf raw values are already in meters:")
        print(f"  Mean altitude: {np.mean(altitudes_direct_km):.1f} km")
        
    else:
        # Normal case - positions in km
        altitudes = radii - R_EARTH/1000
        print(f"\nAltitudes (km above Earth):")
        print(f"  Mean: {np.mean(altitudes):.1f} km")
        print(f"  Std:  {np.std(altitudes):.1f} km")
        print(f"  Min:  {np.min(altitudes):.1f} km") 
        print(f"  Max:  {np.max(altitudes):.1f} km")
    
    # Sample some measurements and validate as orbits
    print("\n=== Validating sample measurements ===")
    sample_indices = [0, len(df)//4, len(df)//2, 3*len(df)//4, -1]
    
    for idx in sample_indices[:3]:  # Just check first 3
        # Try different unit assumptions
        print(f"\nMeasurement {idx}:")
        pos = pos_data[idx]
        vel = vel_data[idx]
        
        # Assumption 1: pos in km, vel in dm/s
        state1 = np.concatenate([pos * 1000, vel * 0.1])
        val1 = validate_orbit_parameters(state1)
        print(f"  If pos[km], vel[dm/s]: alt={val1['altitude_km']:.1f} km, valid={val1['is_valid']}")
        
        # Assumption 2: pos in m, vel in m/s  
        state2 = np.concatenate([pos, vel])
        val2 = validate_orbit_parameters(state2)
        print(f"  If pos[m], vel[m/s]: alt={val2['altitude_km']:.1f} km, valid={val2['is_valid']}")
        
        # Assumption 3: pos in m, vel in dm/s
        state3 = np.concatenate([pos, vel * 0.1])
        val3 = validate_orbit_parameters(state3)
        print(f"  If pos[m], vel[dm/s]: alt={val3['altitude_km']:.1f} km, valid={val3['is_valid']}")
    
    # Plot histogram of radii
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Radius distribution
    ax1.hist(radii, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(R_EARTH/1000, color='red', linestyle='--', label='Earth radius')
    ax1.set_xlabel('Radius (raw units)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of orbital radii')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Velocity magnitude distribution
    vel_mags = np.linalg.norm(vel_data, axis=1)
    ax2.hist(vel_mags, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(76.7, color='red', linestyle='--', label='LEO velocity (if dm/s)')
    ax2.axvline(7670, color='green', linestyle='--', label='LEO velocity (if m/s)')
    ax2.set_xlabel('Velocity magnitude (raw units)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of velocity magnitudes')
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('gps_data_analysis.png', dpi=150)
    print("\nSaved analysis plot to gps_data_analysis.png")
    
    # Return the most likely unit interpretation
    return {
        'position_units': 'meters',  # Based on the huge values
        'velocity_units': 'dm/s',    # Decimters per second
        'mean_altitude_km': (np.mean(radii) - R_EARTH) / 1000
    }

if __name__ == "__main__":
    result = analyze_gps_data()