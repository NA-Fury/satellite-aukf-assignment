<!-- CI status (GitHub Actions) -->
![CI](https://github.com/NA-Fury/satellite-aukf-assignment/actions/workflows/ci.yml/badge.svg)

<!-- License -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# Enhanced Adaptive Unscented Kalman Filter for Satellite Tracking

## 🚀 Project Overview

This repository contains an enhanced implementation of an Adaptive Unscented Kalman Filter (AUKF) for tracking the SWARM-A satellite using GNSS measurements. The implementation features state-of-the-art adaptive filtering techniques, robust data preprocessing, and comprehensive performance analysis.

### Key Features

- **Multiple Adaptive Methods**: Sage-Husa (primary), Innovation-based, and Variational Bayes
- **High-Fidelity Propagation**: Orekit integration with perturbations (J2, drag, SRP)
- **Robust Preprocessing**: Outlier detection and interpolation
- **Comprehensive Analysis**: NIS tests, innovation analysis, 3D visualization
- **Production-Ready**: Type hints, logging, error handling, unit tests

## 📊 Performance Summary

- **Position Accuracy**: ~45m RMSE
- **Velocity Accuracy**: ~0.08 m/s RMSE
- **Processing Rate**: >100 Hz (real-time capable)
- **Adaptation**: Automatic noise covariance tuning
- **Robustness**: Handles 2.3% measurement outliers

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Java 8+ (for Orekit)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/NA-Fury/satellite-aukf-assignment.git
cd satellite-aukf-assignment
```

2. **Create environment (Conda recommended)**
```bash
conda env create -f environment.yml
conda activate satellite-aukf
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. **Download Orekit data**
```bash
# Download orekit-data.zip from https://gitlab.orekit.org/orekit/orekit-data
# Extract to project root directory
wget https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip
unzip orekit-data-master.zip
mv orekit-data-master orekit-data
```

4. **Verify installation**
```bash
python -c "import orekit; orekit.initVM(); print('Orekit OK')"
pytest -v
```

## 📁 Project Structure

```
satellite-aukf-assignment/
├── aukf.py                    # Enhanced AUKF implementation
├── utils.py                   # Utilities (propagator, transforms, tuning)
├── test_aukf.py              # Comprehensive unit tests
├── notebooks/
│   ├── aukf_satellite_tracking.ipynb  # Main implementation notebook
│   └── motion_model.ipynb            # Motion model comparison
├── data/
│   ├── GPS_measurements.parquet     # Raw GNSS data
│   └── GPS_clean.parquet           # Preprocessed data (generated)
├── results/                         # Output directory (generated)
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment
└── README.md                       # This file
```

## 🚀 Quick Start

### Run the main AUKF implementation:
```bash
jupyter lab notebooks/aukf_satellite_tracking.ipynb
```

### Run motion model comparison:
```bash
jupyter lab notebooks/motion_model.ipynb
```

### Run unit tests:
```bash
pytest -v test_aukf.py
```

## 📈 Algorithm Details

### Adaptive Unscented Kalman Filter

The AUKF implementation features:

1. **Sigma Point Generation**
   - Scaled unscented transformation
   - SVD fallback for numerical stability
   - Configurable spreading parameters (α, β, κ)

2. **Adaptive Methods**
   - **Sage-Husa**: Exponential forgetting with dual noise adaptation
   - **Innovation-based**: Windowed covariance estimation
   - **Variational Bayes**: Simplified VB approximation

3. **Motion Models**
   - High-fidelity Orekit propagator (primary)
   - Two-body Keplerian (fallback)
   - Constant velocity (baseline)

### Data Processing Pipeline

1. **Outlier Detection**
   - Position/velocity jump detection
   - Statistical outlier identification
   - Satellite-specific analysis

2. **Coordinate Transformations**
   - ECEF ↔ ECI conversions
   - Time-synchronized transformations
   - IERS conventions compliance

3. **Interpolation**
   - Cubic spline for gap filling
   - Maintains C² continuity
   - Preserves orbital dynamics

## 📊 Results Analysis

The implementation provides comprehensive analysis tools:

### 1. State Estimation Plots
- Position/velocity errors over time
- 3σ uncertainty bounds
- Error distributions

### 2. Filter Consistency
- Normalized Innovation Squared (NIS)
- χ² hypothesis testing
- Innovation whiteness tests

### 3. Adaptive Performance
- Noise covariance evolution
- Convergence analysis
- Adaptation metrics

### 4. 3D Visualization
- ECI trajectory comparison
- Ground track projection
- Altitude profiles

## 🔧 Configuration

### Filter Parameters (aukf.py)

```python
aukf_params = AUKFParameters(
    alpha=1e-3,              # Sigma point spread
    beta=2.0,                # Prior knowledge
    kappa=0.0,               # Secondary scaling
    adaptive_method=AdaptiveMethod.SAGE_HUSA,
    innovation_window=20,    # Window size
    forgetting_factor=0.98,  # Adaptation rate
)
```

### Propagator Settings (utils.py)

```python
propagator = OrbitPropagator(
    use_high_fidelity=True,
    gravity_degree=10,       # EGM degree
    gravity_order=10,        # EGM order
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest -v

# Specific test module
pytest -v test_aukf.py::TestAUKF::test_sigma_point_generation

# With coverage
pytest --cov=aukf --cov=utils -v
```

## 📚 Dependencies

### Core Requirements
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- orekit >= 11.0

### Additional Tools
- pytest >= 6.0 (testing)
- jupyter >= 1.0 (notebooks)
- seaborn >= 0.11 (visualization)
- statsmodels >= 0.12 (analysis)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Orekit team for the excellent orbit propagation library
- Assignment creators for the challenging problem
- Open-source community for various tools and libraries

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Note**: This is an enhanced implementation based on the original assignment requirements. All core algorithms and analysis were developed independently, with AI assistance limited to debugging and code formatting as documented in the notebooks.