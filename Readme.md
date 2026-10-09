# Weather forecasting with ARIMA

This project analyzes and forecasts monthly city temperatures with ARIMA and
seasonal ARIMA (SARIMA), and records experiment results in SQLite.

## Installation

Clone the repository and install its dependencies:

```sh
git clone https://github.com/joas165/WeatherCast-ARIMA.git
cd WeatherCast-ARIMA
pip install -r requirements.txt
```

Download `GlobalLandTemperaturesByMajorCity.csv` from
[Berkeley Earth](http://berkeleyearth.org/data/) and put it in the repository's
root-level `data/` directory. Refer to the dataset's licensing terms before use.

## Run

From the repository root, run:

```sh
python WeatherProject/src/weather_project.py
```

The analysis uses London data from January 2000 through December 2012. It
requires a complete monthly series in that period and stops with an error if
months are missing. Evaluation uses five expanding-window rolling-origin
splits. Each split forecasts 12 months, and the training window expands by
12 months between splits; at least 120 monthly observations are required.
Mean MSE and MAE across the five splits are reported for ARIMA, a seasonal-naive
forecast (the previous year's monthly values), and a SARIMA model with a
12-month seasonal period. The saved ARIMA model is refit on the full available
series after evaluation.

The SQLite database is created at `data/weatherDatabase.db`. Fitted ARIMA
models are saved under `models/`. Each database result records the model
metrics and settings, the saved model path, and SHA-256 hashes of the script
and source dataset so an experiment's inputs can be identified. The recorded
training dates refer to the first split; test dates span all evaluation splits.

## License

This project is licensed under the MIT License.
