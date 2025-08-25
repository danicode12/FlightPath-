# Aircraft Trajectory Prediction (comparing EKF vs LSTM) adding weather soon

**Comparing physics-based (EKF) vs data driven (LSTM) models for short horizon aircraft trajectory prediction**

I set out to explore different prediction methods for aircraft trajectories in the next 60 to 300 seconds. Decided to compare two models that are used in industry such as the extended kalman filter and a LSTM. 

## 🎯 The Big Question

> *What's a more accurate model? LSTM or EKF? Does prediction distance affect this? Does weather affect this?*

I'm testing this by implementing both EKF and modern LSTM neural network, with and without weather data and for different times, to see what actually works best. So far I've been able to build my EKF and test it without weather data but more is yet to come.

## 📊 The Data

Using **~1M real ADS-B position reports** from OpenSky Network for June 12, 2017:
- 5,492 continuous flight segments
- ~10 second update rate
- Full 3D trajectories (lat/lon/alt → East/North/Up)
- Kinematics computed from positions (velocities, turn rates)
- The data was not as good as I expected it to be

## What's Inside of this Mysterious Repo

### Not enough Notebooks

1. **`01_prepare_data.ipynb`** - Data Engineering Pipeline
   - Cleans raw ADS-B data (1.5M → >1M quality records)
   - Segments flights by time gaps and ICOA24
   - Converts global coordinates to local ENU
   - Computes velocities and turn rates
   - Quality filtering (removes crazy altitude changes, unreal speeds, etc.)

2. **`02_ekf.ipynb`** - Finding the Best Flights
   - Tests EKF on candidate flights
   - Identifies TOP 5 best-performing flights
   - Establishes baseline performance metrics
   - Saves results for visualization

3. **`04_visualization.ipynb`** - Beautiful 3D Visualizations
   - 3D trajectory plots with predictions
   - Multi-view analysis (top, side, perspective)
   - Error evolution along flight paths
   - Performance comparison dashboards

### Source Code for basic functions

- **`src/ekf_model.py`** - Coordinated Turn EKF Implementation
  - 7D state vector: [E, N, U, vE, vN, vU, omega]
  - Optional wind compensation
  - Proper Jacobians for linearization
  
- **`src/geo.py`** - Coordinate transformations
  - WGS84 ↔ ENU conversions
  - Local tangent plane projections

- **`src/evaluation.py`** - Metrics
  - RMSE, ADE, FDE calculations
