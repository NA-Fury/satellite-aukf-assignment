# Satellite UKF Assignment

**Author**: NA-Fury  
**Date**: July 2025  

A demonstration pipeline that  
1. **Preprocesses** raw GPS measurements (`GPS_measurements.parquet`) into a cleaned, wide-format table (`GPS_clean.parquet`)  
2. **Applies** an Unscented Kalman Filter (UKF) to fuse multi-satellite ECEF position + velocity measurements  
3. **Diagnoses** filter consistency via a Normalised Innovation Squared (NIS) test  
4. **Compares** a constant-velocity model vs. an analytic two-body propagator  

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
