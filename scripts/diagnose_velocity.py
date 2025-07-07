"""
Diagnose velocity units in GPS data
===================================
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_velocity_units():
    """Diagnose the velocity unit issue."""
    
    # Load data
    df = pd.read_parquet("GPS_clean.parquet")
    cols = json.load(open("meas_cols.json"))
    
    pos_cols = cols[:3]
    vel_cols = cols[3:]
    
    # Get sample of data
    n_samples = min(1000, len(df))
    indices = np.linspace(0, len(df)-1, n_samples, dtype=int)
    
    print("Analyzing velocity units...")
    print("=" * 60)
    
    # For each sample, calculate expected velocity from position
    results = []
    
    for i in indices[:100]:  # Just check first 100
        # Get position in different unit assumptions
        pos_raw = df[pos_cols].iloc[i].values
        vel_raw = df[vel_cols].iloc[i].values
        
        # Calculate orbital radius
        r_raw = np.linalg.norm(pos_raw)
        
        # If positions are in km (which seems correct from r~6850)
        r_m = r_raw * 1000  # Convert to meters
        
        # Expected orbital velocity from vis-viva equation
        mu = 3.986004418e14  # m^3/s^2
        v_expected_ms = np.sqrt(mu / r_m)  # m/s
        
        # Actual velocity magnitude
        v_raw = np.linalg.norm(vel_raw)
        
        # Test different unit assumptions
        result = {
            'r_km': r_raw,
            'v_expected_ms': v_expected_ms,
            'v_raw': v_raw,
            'v_if_dms': v_raw * 0.1,  # If raw is dm/s
            'v_if_ms': v_raw,         # If raw is m/s  
            'v_if_kms': v_raw * 1000, # If raw is km/s
            'scale_to_ms': v_expected_ms / v_raw
        }
        results.append(result)
    
    # Analyze results
    df_results = pd.DataFrame(results)
    
    print(f"\nExpected velocity for orbit at {df_results['r_km'].mean():.1f} km:")
    print(f"  {df_results['v_expected_ms'].mean():.1f} m/s")
    print(f"  {df_results['v_expected_ms'].mean()*10:.1f} dm/s")
    
    print(f"\nActual velocity magnitude (raw units):")
    print(f"  {df_results['v_raw'].mean():.1f} ± {df_results['v_raw'].std():.1f}")
    
    print(f"\nIf velocities are in different units:")
    print(f"  dm/s → m/s: {df_results['v_if_dms'].mean():.1f} m/s")
    print(f"  m/s → m/s:  {df_results['v_if_ms'].mean():.1f} m/s")
    print(f"  km/s → m/s: {df_results['v_if_kms'].mean():.1f} m/s")
    
    print(f"\nRequired scale factor to match expected velocity:")
    print(f"  {df_results['scale_to_ms'].mean():.2f} ± {df_results['scale_to_ms'].std():.2f}")
    
    # The scale factor tells us the unit conversion needed
    scale = df_results['scale_to_ms'].mean()
    if 9.5 < scale < 10.5:
        print("\n✓ Velocities appear to be in m/s, need to multiply by 10 for dm/s")
    elif 0.95 < scale < 1.05:
        print("\n✓ Velocities are already in correct units")
    else:
        print(f"\n⚠️  Unusual scale factor: {scale:.2f}")
        print("   Velocities may be in unexpected units")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Velocity magnitude distribution
    ax = axes[0, 0]
    ax.hist(df[vel_cols].values.flatten(), bins=50, alpha=0.7)
    ax.set_xlabel('Velocity component (raw units)')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Component Distribution')
    ax.set_yscale('log')
    
    # Plot 2: Velocity magnitude vs radius
    ax = axes[0, 1]
    radii = np.linalg.norm(df[pos_cols].values, axis=1)
    vel_mags = np.linalg.norm(df[vel_cols].values, axis=1)
    
    # Sample for plotting
    sample_idx = np.random.choice(len(radii), min(1000, len(radii)), replace=False)
    ax.scatter(radii[sample_idx], vel_mags[sample_idx], alpha=0.5, s=1)
    
    # Add theoretical curve
    r_range = np.linspace(radii.min(), radii.max(), 100)
    v_theoretical = np.sqrt(3.986004418e14 / (r_range * 1000)) * 10  # m/s to dm/s
    ax.plot(r_range, v_theoretical, 'r-', label='Theoretical (if dm/s)', linewidth=2)
    ax.plot(r_range, v_theoretical/10, 'g-', label='Theoretical (if m/s)', linewidth=2)
    ax.plot(r_range, v_theoretical/100, 'b-', label='Theoretical (if 0.1 m/s)', linewidth=2)
    
    ax.set_xlabel('Orbital radius (km)')
    ax.set_ylabel('Velocity magnitude (raw units)')
    ax.set_title('Velocity vs Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scale factor distribution
    ax = axes[1, 0]
    ax.hist(df_results['scale_to_ms'], bins=50, alpha=0.7)
    ax.axvline(10, color='r', linestyle='--', label='10x (m/s → dm/s)')
    ax.axvline(1, color='g', linestyle='--', label='1x (already correct)')
    ax.set_xlabel('Scale factor to match expected velocity')
    ax.set_ylabel('Count')
    ax.set_title('Required Velocity Scale Factor')
    ax.legend()
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""Velocity Unit Diagnosis:
    
Raw velocity magnitude: {vel_mags.mean():.1f} ± {vel_mags.std():.1f}
Expected velocity (m/s): {df_results['v_expected_ms'].mean():.1f}
Expected velocity (dm/s): {df_results['v_expected_ms'].mean()*10:.1f}

Scale factor needed: {scale:.2f}

Conclusion: Velocities appear to be in """
    
    if 9.5 < scale < 10.5:
        summary_text += "m/s\n→ Need to multiply by 10 for dm/s"
    elif 0.95 < scale < 1.05:
        summary_text += "dm/s\n→ Already in correct units"
    else:
        summary_text += f"unknown units\n→ Scale factor: {scale:.2f}"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('velocity_diagnosis.png', dpi=150)
    print(f"\nSaved diagnostic plot to velocity_diagnosis.png")
    
    return scale


def create_fixed_parquet(scale_factor):
    """Create a fixed parquet file with correct units."""
    
    print(f"\nCreating fixed parquet file with velocity scale factor: {scale_factor}")
    
    # Load original data
    df = pd.read_parquet("GPS_clean.parquet")
    cols = json.load(open("meas_cols.json"))
    
    # Create fixed version
    df_fixed = df.copy()
    
    # Positions are already in km (correct)
    # Velocities need scaling
    vel_cols = cols[3:]
    for col in vel_cols:
        df_fixed[col] = df[col] * scale_factor
    
    # Save
    df_fixed.to_parquet("GPS_clean_units_fixed.parquet")
    print("✅ Saved GPS_clean_units_fixed.parquet")
    
    # Save updated column info
    cols_info = {
        "columns": cols,
        "units": {
            "position": "km",
            "velocity": "dm/s"
        },
        "corrections_applied": {
            "position_scale": 1.0,
            "velocity_scale": scale_factor
        }
    }
    
    with open("meas_cols_fixed.json", "w") as f:
        json.dump(cols_info, f, indent=2)
    print("✅ Saved meas_cols_fixed.json")
    
    # Verify a sample
    print("\nVerifying fixed data:")
    pos_km = df_fixed[cols[:3]].iloc[0].values
    vel_dms = df_fixed[cols[3:]].iloc[0].values
    
    # Convert to SI
    state_si = np.concatenate([pos_km * 1000, vel_dms * 0.1])
    
    # Calculate orbital parameters
    r_m = np.linalg.norm(state_si[:3])
    v_ms = np.linalg.norm(state_si[3:])
    alt_km = (r_m - 6.371e6) / 1000
    
    # Period from vis-viva
    mu = 3.986004418e14
    a = r_m  # Assume circular
    period_min = 2 * np.pi * np.sqrt(a**3 / mu) / 60
    
    print(f"  Altitude: {alt_km:.1f} km")
    print(f"  Velocity: {v_ms:.1f} m/s")
    print(f"  Period: {period_min:.1f} minutes")
    
    if 200 < alt_km < 2000 and 7000 < v_ms < 8000:
        print("  ✓ Orbit parameters look correct!")
    else:
        print("  ⚠️  Orbit parameters still seem wrong")
    
    return df_fixed


if __name__ == "__main__":
    # Diagnose the issue
    scale = diagnose_velocity_units()
    
    # If scale factor is close to 10, fix the data
    if 9 < scale < 11:
        print("\n" + "="*60)
        print("DIAGNOSIS: Velocities are in m/s but should be in dm/s")
        print("SOLUTION: Multiply velocities by 10")
        print("="*60)
        
        df_fixed = create_fixed_parquet(10.0)
        
        print("\n✅ Data fixed! Use GPS_clean_units_fixed.parquet for processing")