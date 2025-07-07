"""
Fix GPS data units and create properly scaled parquet file
=========================================================
This script analyzes the GPS data and creates a corrected version
with proper units for satellite orbit determination.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import validate_orbit_parameters, R_EARTH

def fix_gps_units():
    """Analyze and fix GPS data units."""
    
    # Load original data
    print("Loading GPS data...")
    df_orig = pd.read_parquet("GPS_clean.parquet")
    cols = json.load(open("meas_cols.json"))
    
    print(f"Loaded {len(df_orig):,} measurements")
    print(f"Columns: {cols}")
    
    # Create a copy to work with
    df = df_orig.copy()
    
    # Extract data
    pos_cols = cols[:3]
    vel_cols = cols[3:]
    
    # Analyze current values
    print("\n=== Analyzing raw data ===")
    for col in pos_cols:
        values = df[col].values
        print(f"{col}: mean={np.mean(values):.2e}, std={np.std(values):.2e}")
    
    for col in vel_cols:
        values = df[col].values
        print(f"{col}: mean={np.mean(values):.2e}, std={np.std(values):.2e}")
    
    # Calculate orbital parameters with different unit assumptions
    print("\n=== Testing unit conversions ===")
    
    # Sample a few measurements
    sample_indices = [0, len(df)//2, -1]
    
    for idx in sample_indices:
        pos_raw = df[pos_cols].iloc[idx].values
        vel_raw = df[vel_cols].iloc[idx].values
        
        print(f"\nMeasurement {idx}:")
        print(f"  Raw position: {pos_raw}")
        print(f"  Raw velocity: {vel_raw}")
        
        # Test 1: pos already in meters, vel in dm/s
        state1 = np.concatenate([pos_raw, vel_raw * 0.1])
        val1 = validate_orbit_parameters(state1)
        print(f"\n  Test 1 (pos[m], vel[dm/s → m/s]):")
        print(f"    Altitude: {val1['altitude_km']:.1f} km")
        print(f"    Velocity: {val1['velocity_m_s']:.1f} m/s")
        print(f"    Period: {val1['period_minutes']:.1f} min")
        print(f"    Valid: {val1['is_valid']}")
        
        # Test 2: Both need scaling down by 1000
        state2 = np.concatenate([pos_raw/1000, vel_raw/10])
        val2 = validate_orbit_parameters(state2)
        print(f"\n  Test 2 (pos[?→m]/1000, vel[?→m/s]/10):")
        print(f"    Altitude: {val2['altitude_km']:.1f} km")
        print(f"    Valid: {val2['is_valid']}")
    
    # Based on the huge position values, they're likely in meters
    # The filter expects km and dm/s, so we need to convert
    print("\n=== Applying unit corrections ===")
    print("Converting: positions from m to km, velocities from dm/s to dm/s (no change)")
    
    # Apply conversions
    df_fixed = df.copy()
    
    # If positions are in meters, convert to km
    if np.mean(np.abs(df[pos_cols].values)) > 1e6:  # Likely in meters
        print("✓ Converting positions from meters to kilometers")
        for col in pos_cols:
            df_fixed[col] = df[col] / 1000.0
    
    # Velocities appear to already be in dm/s (based on typical values ~70000)
    # which would be ~7000 m/s, appropriate for LEO
    print("✓ Keeping velocities in dm/s (already correct)")
    
    # Verify the fix
    print("\n=== Verifying corrected data ===")
    for idx in [0, len(df)//2, -1]:
        pos_km = df_fixed[pos_cols].iloc[idx].values
        vel_dms = df_fixed[vel_cols].iloc[idx].values
        
        # Convert to SI for validation
        state_si = np.concatenate([pos_km * 1000, vel_dms * 0.1])
        val = validate_orbit_parameters(state_si)
        
        print(f"\nCorrected measurement {idx}:")
        print(f"  Altitude: {val['altitude_km']:.1f} km")
        print(f"  Velocity: {val['velocity_m_s']:.1f} m/s")
        print(f"  Period: {val['period_minutes']:.1f} min")
        print(f"  Valid: {val['is_valid']}")
    
    # Save corrected data
    output_file = "GPS_clean_fixed_units.parquet"
    df_fixed.to_parquet(output_file)
    print(f"\n✅ Saved corrected data to {output_file}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original vs fixed position magnitudes
    r_orig = np.linalg.norm(df[pos_cols].values, axis=1)
    r_fixed = np.linalg.norm(df_fixed[pos_cols].values, axis=1)
    
    ax1.hist(r_orig, bins=50, alpha=0.7, label='Original', color='red')
    ax1.set_xlabel('Radius (original units)')
    ax1.set_ylabel('Count')
    ax1.set_title('Original Position Magnitudes')
    ax1.set_yscale('log')
    
    ax2.hist(r_fixed, bins=50, alpha=0.7, label='Fixed', color='green')
    ax2.axvline(R_EARTH/1000, color='red', linestyle='--', label='Earth radius')
    ax2.set_xlabel('Radius (km)')
    ax2.set_ylabel('Count')
    ax2.set_title('Fixed Position Magnitudes')
    ax2.set_yscale('log')
    ax2.legend()
    
    # Altitude distribution
    altitudes = r_fixed - R_EARTH/1000
    ax3.hist(altitudes, bins=50, alpha=0.7, color='blue')
    ax3.set_xlabel('Altitude (km)')
    ax3.set_ylabel('Count')
    ax3.set_title('Orbital Altitudes (Fixed Data)')
    ax3.axvline(400, color='red', linestyle='--', label='ISS altitude')
    ax3.legend()
    
    # Velocity magnitudes
    v_mags = np.linalg.norm(df_fixed[vel_cols].values * 0.1, axis=1)  # Convert to m/s
    ax4.hist(v_mags, bins=50, alpha=0.7, color='purple')
    ax4.set_xlabel('Velocity (m/s)')
    ax4.set_ylabel('Count')
    ax4.set_title('Velocity Magnitudes')
    ax4.axvline(7670, color='red', linestyle='--', label='Circular LEO')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('unit_correction_analysis.png', dpi=150)
    print("✅ Saved analysis plot to unit_correction_analysis.png")
    
    # Summary statistics
    print("\n=== Summary Statistics (Fixed Data) ===")
    print(f"Mean altitude: {np.mean(altitudes):.1f} km")
    print(f"Altitude range: {np.min(altitudes):.1f} to {np.max(altitudes):.1f} km")
    print(f"Mean velocity: {np.mean(v_mags):.1f} m/s")
    print(f"Velocity range: {np.min(v_mags):.1f} to {np.max(v_mags):.1f} m/s")
    
    # Update meas_cols.json with unit information
    cols_with_units = {
        "columns": cols,
        "units": {
            "position": "km",
            "velocity": "dm/s"
        },
        "notes": "Fixed from original data where positions were in meters"
    }
    
    with open("meas_cols_with_units.json", "w") as f:
        json.dump(cols_with_units, f, indent=2)
    print("\n✅ Saved column metadata to meas_cols_with_units.json")
    
    return df_fixed

if __name__ == "__main__":
    df_fixed = fix_gps_units()