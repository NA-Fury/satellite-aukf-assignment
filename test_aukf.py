import unittest
import numpy as np
from aukf import AdaptiveUnscentedKalmanFilter, AUKFParameters, AdaptiveMethod
import warnings


class TestAUKF(unittest.TestCase):
    """Unit tests for Adaptive Unscented Kalman Filter implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 4  # Simple 2D position-velocity model
        self.measurement_dim = 2  # Position only measurements
        self.dt = 1.0
        
        # Create filter instance
        self.params = AUKFParameters(
            alpha=1e-3,
            beta=2.0,
            kappa=0.0,
            adaptive_method=AdaptiveMethod.SAGE_HUSA
        )
        
        self.filter = AdaptiveUnscentedKalmanFilter(
            state_dim=self.state_dim,
            measurement_dim=self.measurement_dim,
            dt=self.dt,
            params=self.params
        )
        
        # Set initial conditions
        self.x0 = np.array([0.0, 0.0, 1.0, 0.5])  # [x, y, vx, vy]
        self.P0 = np.eye(4) * 0.1
        self.Q0 = np.eye(4) * 0.01
        self.R0 = np.eye(2) * 0.05
        
        self.filter.set_initial_conditions(self.x0, self.P0, self.Q0, self.R0)
    
    def test_initialization(self):
        """Test filter initialization"""
        # Check dimensions
        self.assertEqual(self.filter.state_dim, self.state_dim)
        self.assertEqual(self.filter.measurement_dim, self.measurement_dim)
        
        # Check initial state
        np.testing.assert_array_equal(self.filter.x, self.x0)
        np.testing.assert_array_equal(self.filter.P, self.P0)
        
        # Check weights sum to 1
        self.assertAlmostEqual(np.sum(self.filter.Wm), 1.0, places=10)
        self.assertAlmostEqual(np.sum(self.filter.Wc), 1.0, places=10)
    
    def test_sigma_point_generation(self):
        """Test sigma point generation with known inputs"""
        # Simple test case: identity covariance
        x = np.array([1.0, 2.0, 3.0, 4.0])
        P = np.eye(4)
        
        sigma_points = self.filter.generate_sigma_points(x, P)
        
        # Check dimensions
        self.assertEqual(sigma_points.shape, (4, 9))  # 2n+1 sigma points
        
        # Check mean preservation
        mean_reconstructed = np.sum(self.filter.Wm * sigma_points, axis=1)
        np.testing.assert_array_almost_equal(mean_reconstructed, x, decimal=10)
        
        # Check covariance preservation
        cov_reconstructed = np.zeros((4, 4))
        for i in range(9):
            y = sigma_points[:, i] - x
            cov_reconstructed += self.filter.Wc[i] * np.outer(y, y)
        np.testing.assert_array_almost_equal(cov_reconstructed, P, decimal=10)
    
    def test_predict_step(self):
        """Test prediction step with linear motion model"""
        # Define simple constant velocity motion model
        def motion_model(x, dt, u=None):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return F @ x
        
        # Store initial state
        x_init = self.filter.x.copy()
        
        # Predict
        self.filter.predict(motion_model)
        
        # For linear model, prediction should match analytical result
        x_expected = motion_model(x_init, self.dt)
        np.testing.assert_array_almost_equal(self.filter.x, x_expected, decimal=10)
        
        # Check covariance increased (due to process noise)
        self.assertTrue(np.trace(self.filter.P) > np.trace(self.P0))
    
    def test_update_step(self):
        """Test update step with linear measurement model"""
        # Define measurement model (observe position only)
        def measurement_model(x):
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            return H @ x
        
        # Simple motion model for prediction
        def motion_model(x, dt, u=None):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return F @ x
        
        # Predict first
        self.filter.predict(motion_model)
        x_pred = self.filter.x.copy()
        P_pred = self.filter.P.copy()
        
        # Create measurement
        z_true = measurement_model(x_pred)
        z_noisy = z_true + np.random.multivariate_normal([0, 0], self.R0)
        
        # Update
        self.filter.update(z_noisy, measurement_model)
        
        # Check that update moved estimate toward measurement
        innovation = z_noisy - measurement_model(x_pred)
        x_change = self.filter.x - x_pred
        
        # Position components should move in direction of innovation
        self.assertTrue(np.sign(x_change[0]) == np.sign(innovation[0]) or 
                       abs(x_change[0]) < 1e-10)
        self.assertTrue(np.sign(x_change[1]) == np.sign(innovation[1]) or 
                       abs(x_change[1]) < 1e-10)
        
        # Covariance should decrease after update
        self.assertTrue(np.trace(self.filter.P) < np.trace(P_pred))
    
    def test_full_filter_cycle(self):
        """Test complete filter cycle with synthetic data"""
        # Generate synthetic trajectory
        n_steps = 50
        true_states = []
        measurements = []
        
        # Initial true state
        x_true = np.array([0.0, 0.0, 1.0, 0.5])
        
        # Motion and measurement models
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        def motion_model(x, dt, u=None):
            return F @ x + np.random.multivariate_normal(np.zeros(4), self.Q0)
        
        def measurement_model(x):
            return H @ x
        
        # Generate data
        for i in range(n_steps):
            x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), self.Q0 * 0.1)
            true_states.append(x_true)
            
            z = H @ x_true + np.random.multivariate_normal(np.zeros(2), self.R0)
            measurements.append(z)
        
        # Run filter
        estimates = []
        errors = []
        
        for i in range(n_steps):
            # Predict
            self.filter.predict(lambda x, dt, u: F @ x)  # Use noiseless model for prediction
            
            # Update
            self.filter.update(measurements[i], measurement_model)
            
            # Store results
            estimates.append(self.filter.x.copy())
            errors.append(np.linalg.norm(self.filter.x - true_states[i]))
        
        # Check convergence
        errors = np.array(errors)
        
        # Average error should be reasonable
        self.assertLess(np.mean(errors), 1.0)
        
        # Error should stabilize (last half should have lower variance than first half)
        first_half_var = np.var(errors[:n_steps//2])
        second_half_var = np.var(errors[n_steps//2:])
        self.assertLess(second_half_var, first_half_var * 1.5)  # Allow some margin
    
    def test_adaptive_noise_estimation(self):
        """Test that adaptive algorithm adjusts noise covariances"""
        # Store initial noise estimates
        Q_initial = self.filter.Q.copy()
        R_initial = self.filter.R.copy()
        
        # Run filter with measurements that have different noise characteristics
        def motion_model(x, dt, u=None):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return F @ x
        
        def measurement_model(x):
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            return H @ x
        
        # Generate measurements with higher noise than initialized
        high_noise_R = self.R0 * 5
        
        for i in range(30):  # Need enough iterations for adaptation
            self.filter.predict(motion_model)
            
            # Create measurement with higher noise
            z_true = measurement_model(self.filter.x)
            z_noisy = z_true + np.random.multivariate_normal([0, 0], high_noise_R)
            
            self.filter.update(z_noisy, measurement_model)
        
        # Check that noise estimates have adapted
        Q_final = self.filter.Q
        R_final = self.filter.R
        
        # At least one of the noise covariances should have changed
        Q_changed = not np.allclose(Q_initial, Q_final)
        R_changed = not np.allclose(R_initial, R_final)
        
        self.assertTrue(Q_changed or R_changed, 
                       "Adaptive algorithm should adjust noise estimates")
    
    def test_numerical_stability(self):
        """Test filter stability with challenging conditions"""
        # Test with near-singular covariance
        P_singular = np.eye(4) * 1e-12
        P_singular[0, 1] = 1e-6  # Add correlation
        P_singular[1, 0] = 1e-6
        
        # Should not raise error
        try:
            sigma_points = self.filter.generate_sigma_points(self.x0, P_singular)
            self.assertEqual(sigma_points.shape, (4, 9))
        except Exception as e:
            self.fail(f"Sigma point generation failed with near-singular P: {e}")
        
        # Test with large state values
        x_large = np.array([1e6, 1e6, 1e3, 1e3])
        P_large = np.eye(4) * 1e8
        
        try:
            sigma_points = self.filter.generate_sigma_points(x_large, P_large)
            # Check that sigma points maintain reasonable spread
            spread = np.max(np.abs(sigma_points - x_large[:, np.newaxis]))
            self.assertGreater(spread, 0)  # Should have some spread
            self.assertLess(spread, 1e10)  # But not infinite
        except Exception as e:
            self.fail(f"Sigma point generation failed with large values: {e}")
    
    def test_innovation_statistics(self):
        """Test that innovation sequence has expected statistical properties"""
        # Run filter to accumulate innovations
        def motion_model(x, dt, u=None):
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            return F @ x
        
        def measurement_model(x):
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            return H @ x
        
        # Generate consistent measurements
        for i in range(50):
            self.filter.predict(motion_model)
            
            # Perfect measurements (should lead to zero-mean innovations)
            z = measurement_model(self.filter.x)
            self.filter.update(z, measurement_model)
        
        # Check innovation statistics
        stats = self.filter.get_filter_statistics()
        
        if 'innovation_mean' in stats and stats['innovation_mean'] is not None:
            # Innovation mean should be near zero
            np.testing.assert_array_almost_equal(
                stats['innovation_mean'], 
                np.zeros(self.measurement_dim), 
                decimal=2
            )
        
        # Check NIS is computed
        if 'normalized_innovation_squared' in stats:
            nis = stats['normalized_innovation_squared']
            if nis is not None:
                # NIS should be positive
                self.assertGreater(nis, 0)
    
    def test_reset_functionality(self):
        """Test adaptation history reset"""
        # Add some history
        for i in range(10):
            self.filter.innovation_history.append(np.random.randn(self.measurement_dim))
            self.filter.S_history.append(np.eye(self.measurement_dim))
        
        self.filter.time_step = 10
        
        # Reset
        self.filter.reset_adaptation_history()
        
        # Check reset worked
        self.assertEqual(len(self.filter.innovation_history), 0)
        self.assertEqual(len(self.filter.S_history), 0)
        self.assertEqual(self.filter.time_step, 0)


class TestAUKFEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_different_adaptive_methods(self):
        """Test that all adaptive methods can be instantiated and run"""
        for method in AdaptiveMethod:
            params = AUKFParameters(adaptive_method=method)
            filter = AdaptiveUnscentedKalmanFilter(
                state_dim=2,
                measurement_dim=1,
                dt=1.0,
                params=params
            )
            
            # Should initialize without error
            self.assertEqual(filter.params.adaptive_method, method)
            
            # Should run one cycle without error
            filter.set_initial_conditions(
                np.array([0.0, 1.0]),
                np.eye(2),
                np.eye(2) * 0.01,
                np.array([[0.1]])
            )
            
            filter.predict(lambda x, dt, u: x)
            filter.update(np.array([0.0]), lambda x: np.array([x[0]]))
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches"""
        filter = AdaptiveUnscentedKalmanFilter(
            state_dim=3,
            measurement_dim=2,
            dt=1.0
        )
        
        # Wrong initial state dimension
        with self.assertRaises(Exception):
            filter.set_initial_conditions(
                np.array([1, 2]),  # Wrong size
                np.eye(3),
                np.eye(3),
                np.eye(2)
            )
    
    def test_extreme_parameters(self):
        """Test filter behavior with extreme parameter values"""
        # Very small alpha (sigma points very close to mean)
        params_small_alpha = AUKFParameters(alpha=1e-6)
        filter1 = AdaptiveUnscentedKalmanFilter(2, 1, 1.0, params_small_alpha)
        
        # Very large alpha (sigma points far from mean)
        params_large_alpha = AUKFParameters(alpha=10.0)
        filter2 = AdaptiveUnscentedKalmanFilter(2, 1, 1.0, params_large_alpha)
        
        # Both should initialize without error
        self.assertIsNotNone(filter1)
        self.assertIsNotNone(filter2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)