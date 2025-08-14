# Comparing EKF and LSTM for Short Horizon Trajectory Prediction
The goal is to predict ENU at +60s, +120s, +300s by using ADS-B and METAR/TAF data. 
The idea is simple: It's clear that weather data is important in prediction for long horizon aviation predictions. But can I improve my short-horizon model by introducing weather data? 

## Variants tested here:
1. **EKF** → Physics, no weather.
2. **EKF+WX** → Physics + wind bias.
3. **LSTM** → Data-driven, no weather.
4. **LSTM+WX** → Data-driven + wind, temp, pressure, etc.


## Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook

## Layout
1. Four main notebooks: data processing, EKF Model, LSTM Model, and a comparison.
2. Src: used python modules including the models and some data processing functions
3. Plots and Images
4. Data used - Taken from Opensky Network
