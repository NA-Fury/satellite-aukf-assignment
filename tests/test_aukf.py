"""
Unit tests for Adaptive UKF implementation.
"""

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aukf import AdaptiveUKF, FilterParameters
from src.utils import measurement_model


class TestAdaptiveUKF:
    """Test suite for AdaptiveUKF class"""

    @pytest.fixture
    def simple_motion_model(self):
        """Simple linear motion model for testing"""

        def model(state, dt):
            # Simple constant velocity model
            A = np.array(
                [
                    [1, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ]
            )
            return A @ state

        return model

    @pytest.fixture
    def basic_filter(self, simple_motion_model):
        """Create a basic filter for testing"""
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 1.0
        process_noise = np.eye(6) * 0.1
        measurement_noise = np.eye(6) * 0.5

        return AdaptiveUKF(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            motion_model=simple_motion_model,
            measurement_model=measurement_model,
        )

    def test_initialization(self, basic_filter):
        """Test filter initialization"""
        assert basic_filter.n == 6
        assert basic_filter.x.shape == (6,)
        assert basic_filter.P.shape == (6, 6)
        assert basic_filter.Q.shape == (6, 6)
        assert basic_filter.R.shape == (6, 6)

    def test_sigma_points_generation(self, basic_filter):
        """Test sigma points generation"""
        x = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        P = np.eye(6) * 1.0

        sigma_points = basic_filter.generate_sigma_points(x, P)

        # Check dimensions
        assert sigma_points.shape == (6, 13)

        # Check mean preservation
        weighted_mean = np.sum(basic_filter.Wm[:, np.newaxis] * sigma_points, axis=0)
        assert_allclose(weighted_mean, x, rtol=1e-10)

    def test_predict_step(self, basic_filter):
        """Test prediction step"""
        dt = 10.0
        x_pred, P_pred = basic_filter.predict(dt)

        # Check state propagation
        assert x_pred.shape == (6,)
        assert P_pred.shape == (6, 6)

        # Check that position has changed due to velocity
        assert x_pred[0] != basic_filter.x[0]
        assert_allclose(x_pred[0], 7000.0 + 0.0 * dt, rtol=1e-10)  # x + vx*dt
        assert_allclose(x_pred[1], 0.0 + 7.5 * dt, rtol=1e-10)  # y + vy*dt

    def test_update_step(self, basic_filter):
        """Test update step"""
        # First predict
        dt = 10.0
        basic_filter.predict(dt)

        # Create synthetic measurement
        measurement = np.array([7000.0, 75.0, 0.0, 0.0, 7.5, 0.0])

        # Update
        x_updated, P_updated = basic_filter.update(measurement, 0.0)

        assert x_updated.shape == (6,)
        assert P_updated.shape == (6, 6)

        # Check that update moved state closer to measurement
        error_before = np.linalg.norm(basic_filter.x_pred - measurement)
        error_after = np.linalg.norm(x_updated - measurement)
        assert error_after < error_before

    def test_positive_definite_enforcement(self, basic_filter):
        """Test positive definite matrix enforcement"""
        # Create a non-positive definite matrix
        bad_matrix = np.array([[1, 2], [2, 1]])

        # Fix it
        fixed_matrix = basic_filter._ensure_positive_definite(bad_matrix)

        # Check that it's now positive definite
        eigenvalues = np.linalg.eigvals(fixed_matrix)
        assert np.all(eigenvalues > 0)

    def test_outlier_rejection(self, basic_filter):
        """Test measurement outlier rejection"""
        # Set up filter
        dt = 10.0
        basic_filter.predict(dt)

        # Create outlier measurement (very far from predicted state)
        outlier = np.array([10000.0, 10000.0, 10000.0, 100.0, 100.0, 100.0])

        # Store initial stats
        initial_outliers = basic_filter.filter_stats["outliers_rejected"]

        # Update with outlier
        x_updated, P_updated = basic_filter.update(outlier, 0.0)

        # Check that outlier was rejected
        assert basic_filter.filter_stats["outliers_rejected"] > initial_outliers

    def test_adaptive_noise_estimation(self, basic_filter):
        """Test adaptive noise covariance estimation"""
        # Store initial noise estimates
        Q_initial = basic_filter.Q_k.copy()
        R_initial = basic_filter.R_k.copy()

        # Run multiple predict-update cycles
        for i in range(20):
            basic_filter.predict(1.0)
            measurement = basic_filter.x_pred + np.random.randn(6) * 0.1
            basic_filter.update(measurement, i)

        # Check that noise estimates have adapted
        assert not np.allclose(basic_filter.Q_k, Q_initial)
        assert not np.allclose(basic_filter.R_k, R_initial)


class TestFilterConvergence:
    """Test filter convergence properties"""

    def test_steady_state_convergence(self, simple_motion_model):
        """Test that filter converges to steady state"""
        # Initialize filter with large uncertainty
        initial_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        initial_cov = np.eye(6) * 100.0  # Large initial uncertainty
        process_noise = np.eye(6) * 0.01
        measurement_noise = np.eye(6) * 1.0

        filter = AdaptiveUKF(
            initial_state=initial_state,
            initial_covariance=initial_cov,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            motion_model=simple_motion_model,
            measurement_model=measurement_model,
        )

        # Run filter for many steps with consistent measurements
        trace_history = []
        for i in range(100):
            filter.predict(1.0)
            true_state = initial_state + np.array([0, i * 7.5, 0, 0, 0, 0])
            measurement = true_state + np.random.randn(6) * 0.1
            filter.update(measurement, i)
            trace_history.append(np.trace(filter.P))

        # Check that uncertainty decreases and stabilizes
        assert trace_history[-1] < trace_history[0] * 0.1
        assert np.std(trace_history[-20:]) < 0.1  # Stable in last 20 steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
