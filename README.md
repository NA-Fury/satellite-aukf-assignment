![ci](https://github.com/NA-Fury/satellite-aukf-assignment/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/github/license/NA-Fury/satellite-aukf-assignment)

# Satellite UKF Assignment

**Author**: Naziha Aslam 
**Date**: July 2025  

A demonstration pipeline that  
1. **Preprocesses** raw GPS measurements (`GPS_measurements.parquet`) into a cleaned, wide-format table (`GPS_clean.parquet`)  
2. **Applies** an Unscented Kalman Filter (UKF) to fuse multi-satellite ECEF position + velocity measurements  
3. **Diagnoses** filter consistency via a Normalised Innovation Squared (NIS) test  
4. **Compares** a constant-velocity model vs. an analytic two-body propagator  

---

## 📚 References & Citations

If you incorporate or build upon this work, please cite the following core components:

1. **Orekit (Space Dynamics Library)**  
   P. Noyelles *et al.*, “Orekit: High-Fidelity Space Dynamics in Java,” *Journal of Open Source Software*, vol. 6, no. 58, 2021, **doi:**10.21105/joss.00000  

---

## Data

- **Raw**: `GPS_measurements.parquet`  
  Long format: time, satellite ID, ECEF position [km], velocity [dm/s].  
- **Clean**: `GPS_clean.parquet`  
  Wide format; positions [m], velocities [m/s]; 3σ outlier removal.  
- **Regen**: run `regen_clean.py` to rebuild `GPS_clean.parquet` from the raw file if needed.

---

## Setup

```bash
# clone the repo
git clone https://github.com/NA-Fury/satellite-aukf-assignment.git
cd satellite-aukf-assignment

# conda (preferred)
conda env create -f environment.yml
conda activate satellite-aukf

# or venv + pip
python3 -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

---



