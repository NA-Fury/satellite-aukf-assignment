<!-- CI status (GitHub Actions) -->
![CI](https://github.com/NA-Fury/satellite-aukf-assignment/actions/workflows/ci.yml/badge.svg)

<!-- License -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# 🚀 Satellite UKF Assignment

**Author**: Naziha Aslam  
**Date**: July 2025  

A demonstration pipeline that:

1. 📝 **Preprocesses** raw GPS measurements (`data/GPS_measurements.parquet`) into a cleaned, wide-format table (`GPS_clean.parquet`)  
2. 🔄 **Applies** an Unscented Kalman Filter (UKF) to fuse multi-satellite ECEF position + velocity measurements  
3. 📊 **Diagnoses** filter consistency via a Normalised Innovation Squared (NIS) test  
4. 🌍 **Compares** a constant-velocity model vs. a two-body propagator  

---

## 📚 References & Citations

If you use or build upon this work, please cite:

> Aslam, N. (2025). *Satellite UKF Assignment* (v1.0.0). GitHub. https://github.com/NA-Fury/satellite-aukf-assignment

**Primary algorithm & tools**  
- Julier, S. J. & Uhlmann, J. K. (1997). New extension of the Kalman filter to nonlinear systems. *SPIE*.  
- P. Noyelles *et al.* (2021). “Orekit: High-Fidelity Space Dynamics in Java,” *Journal of Open Source Software*, 6(58). doi:10.21105/joss.00000  

---

## 📂 Data

- **Raw**: `data/GPS_measurements.parquet`  
  • Long format: time, satellite ID, ECEF position [km], velocity [dm/s]  
- **Clean**: `GPS_clean.parquet`  
  • Wide format: positions [m], velocities [m/s]; 3σ outlier removal  
- **Regen**: `regen_clean.py` to rebuild the clean file  

---

## 🛠️ Setup & Usage

```bash
# Clone
git clone https://github.com/NA-Fury/satellite-aukf-assignment.git
cd satellite-aukf-assignment

# Conda (preferred)
conda env create -f environment.yml
conda activate satellite-aukf

# Or venv + pip
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

pip install -e .
pip install -r requirements.txt

# Re-generate clean data
python regen_clean.py

# Run the filter demo (notebook)
jupyter lab notebooks/02_filter_demo.ipynb

# Run unit tests
pytest -q
