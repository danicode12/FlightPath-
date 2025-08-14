# Weather-Aware Short-Horizon Trajectory Prediction (EKF + LSTM)
Predict E,N,U at +60/+120/+300s using ADS-B and METAR/TAF.
Variants: EKF, EKF+WX, LSTM, LSTM+WX. Metrics: RMSE, ADE, FDE.

## Quickstart
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook

## Layout
notebooks/01_prepare_data.ipynb … 04_compare.ipynb
src/ reusable modules; plots/ images; data/ gitignored.

## Results (to fill)
- RMSE table
- Error vs horizon plot
- Windy day case study
