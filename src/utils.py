"""
Utilities for satellite orbit propagation and visualization
==========================================================
Includes a simple but accurate orbit propagator for Earth satellites.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Earth parameters
MU_EARTH = 3.986004418e14  # m^3/s^2 (Earth gravitational parameter)
R_EARTH = 6371000.0        # m (Earth radius)
J2 = 1.08262668e-3         # J2 perturbation coefficient

def propagate_orbit(state, dt):
    """
    Propagate satellite orbit using a simple but accurate model.
    
    Parameters:
    -----------
    state : np.ndarray
        State vector [x, y, z, vx, vy, vz] in meters and m/s
    dt : float
        Time step in seconds
        
    Returns:
    --------
    new_state : np.ndarray
        Propagated state vector
    """
    # Extract position and velocity
    r = state[:3].copy()
    v = state[3:].copy()
    
    # For small time steps, use RK4 integration
    if abs(dt) < 1.0:
        # Single step RK4
        k1_r, k1_v = _orbit_derivatives(r, v)
        k2_r, k2_v = _orbit_derivatives(r + 0.5*dt*k1_r, v + 0.5*dt*k1_v)
        k3_r, k3_v = _orbit_derivatives(r + 0.5*dt*k2_r, v + 0.5*dt*k2_v)
        k4_r, k4_v = _orbit_derivatives(r + dt*k3_r, v + dt*k3_v)
        
        new_r = r + (dt/6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        new_v = v + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    else:
        # For larger time steps, use multiple sub-steps
        n_steps = max(2, int(abs(dt) / 10.0))
        dt_step = dt / n_steps
        
        new_r = r.copy()
        new_v = v.copy()
        
        for _ in range(n_steps):
            k1_r, k1_v = _orbit_derivatives(new_r, new_v)
            k2_r, k2_v = _orbit_derivatives(new_r + 0.5*dt_step*k1_r, new_v + 0.5*dt_step*k1_v)
            k3_r, k3_v = _orbit_derivatives(new_r + 0.5*dt_step*k2_r, new_v + 0.5*dt_step*k2_v)
            k4_r, k4_v = _orbit_derivatives(new_r + dt_step*k3_r, new_v + dt_step*k3_v)
            
            new_r = new_r + (dt_step/6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            new_v = new_v + (dt_step/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return np.concatenate([new_r, new_v])

def _orbit_derivatives(r, v):
    """
    Calculate orbital derivatives including J2 perturbation.
    
    Parameters:
    -----------
    r : np.ndarray
        Position vector [x, y, z] in meters
    v : np.ndarray
        Velocity vector [vx, vy, vz] in m/s
        
    Returns:
    --------
    dr_dt : np.ndarray
        Velocity (derivative of position)
    dv_dt : np.ndarray
        Acceleration (derivative of velocity)
    """
    r_mag = np.linalg.norm(r)
    
    # Two-body acceleration
    a_2body = -MU_EARTH / r_mag**3 * r
    
    # J2 perturbation acceleration
    x, y, z = r
    r2 = r_mag * r_mag
    
    # J2 acceleration components
    factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_mag**5
    
    ax_j2 = factor * x * (5 * z**2 / r2 - 1)
    ay_j2 = factor * y * (5 * z**2 / r2 - 1)
    az_j2 = factor * z * (5 * z**2 / r2 - 3)
    
    a_j2 = np.array([ax_j2, ay_j2, az_j2])
    
    # Total acceleration
    dv_dt = a_2body + a_j2
    dr_dt = v
    
    return dr_dt, dv_dt

def plot_ground_track(positions, title="Satellite Ground Track", save_as=None):
    """
    Plot satellite ground track on Earth.
    
    Parameters:
    -----------
    positions : np.ndarray
        Array of positions in km, shape (n_points, 3)
    title : str
        Plot title
    save_as : str or None
        Filename to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to Earth-centered coordinates
    x_km = positions[:, 0] / 1000
    y_km = positions[:, 1] / 1000
    z_km = positions[:, 2] / 1000
    
    # Calculate latitude and longitude
    r_xy = np.sqrt(x_km**2 + y_km**2)
    lat = np.degrees(np.arctan2(z_km, r_xy))
    lon = np.degrees(np.arctan2(y_km, x_km))
    
    # Plot ground track
    ax.plot(lon, lat, 'b-', linewidth=1.5, alpha=0.8)
    ax.scatter(lon[0], lat[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(lon[-1], lat[-1], c='red', s=100, marker='s', label='End', zorder=5)
    
    # Add Earth coastlines (simplified)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
    
    return fig, ax

def validate_orbit_parameters(state):
    """
    Check if orbit parameters are reasonable for Earth satellites.
    
    Parameters:
    -----------
    state : np.ndarray
        State vector [x, y, z, vx, vy, vz] in SI units
        
    Returns:
    --------
    dict : Dictionary with validation results
    """
    r = state[:3]
    v = state[3:]
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    
    # Calculate orbital elements
    h = np.cross(r, v)  # Angular momentum
    h_mag = np.linalg.norm(h)
    
    # Eccentricity vector
    e_vec = np.cross(v, h) / MU_EARTH - r / r_mag
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis
    a = 1 / (2/r_mag - v_mag**2/MU_EARTH)
    
    # Altitude
    altitude = (r_mag - R_EARTH) / 1000  # km
    
    # Period
    if a > 0:
        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH) / 60  # minutes
    else:
        period = np.nan
    
    return {
        'radius_km': r_mag / 1000,
        'altitude_km': altitude,
        'velocity_m_s': v_mag,
        'semi_major_axis_km': a / 1000,
        'eccentricity': e,
        'period_minutes': period,
        'is_valid': (200 < altitude < 50000) and (e < 0.9) and (3000 < v_mag < 11000)
    }

# Test the propagator
if __name__ == "__main__":
    # Test with ISS-like orbit
    r0 = np.array([R_EARTH + 400e3, 0, 0])  # 400 km altitude
    v0 = np.array([0, 7670, 0])  # Circular orbit velocity
    
    state0 = np.concatenate([r0, v0])
    print("Initial state validation:")
    validation = validate_orbit_parameters(state0)
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Propagate for one orbit
    dt = 10.0  # 10 second steps
    n_steps = int(validation['period_minutes'] * 60 / dt)
    
    states = [state0]
    for i in range(n_steps):
        states.append(propagate_orbit(states[-1], dt))
    
    states = np.array(states)
    
    # Plot orbit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # X-Y projection
    ax1.plot(states[:, 0]/1e6, states[:, 1]/1e6, 'b-')
    earth = Circle((0, 0), R_EARTH/1e6, facecolor='lightblue', edgecolor='blue')
    ax1.add_patch(earth)
    ax1.set_xlabel('X (Mm)')
    ax1.set_ylabel('Y (Mm)')
    ax1.set_title('Orbit in X-Y plane')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Altitude vs time
    altitudes = [validate_orbit_parameters(s)['altitude_km'] for s in states]
    times = np.arange(len(states)) * dt / 60
    ax2.plot(times, altitudes)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Altitude variation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('orbit_propagator_test.png', dpi=150)
    print(f"\nPropagated {len(states)} steps")
    print(f"Final altitude: {altitudes[-1]:.1f} km")
    print(f"Altitude variation: {max(altitudes) - min(altitudes):.2f} km")