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

## Dataset

The dataset is not included in the repository. From the repository root, install
KaggleHub and use it to download the latest version of the **Climate Change:
Earth Surface Temperature Data** dataset from Kaggle:

```powershell
python -m pip install kagglehub
python -c "from pathlib import Path; import shutil, kagglehub; source = Path(kagglehub.dataset_download('berkeleyearth/climate-change-earth-surface-temperature-data')) / 'GlobalLandTemperaturesByMajorCity.csv'; destination = Path('data') / source.name; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination); print('Dataset saved to:', destination.resolve())"
```

This saves `GlobalLandTemperaturesByMajorCity.csv` under the repository's
root-level `data/` directory. KaggleHub may require Kaggle authentication.
The loader needs the columns `dt`, `City`, and
`AverageTemperature`, including complete monthly London data from January 2000
through December 2012. If the file is missing or the needed months are absent,
the analysis will stop with an error.

Kaggle lists this dataset as **CC BY-NC-SA 4.0** on its
[dataset page](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data).
The page says the data is repackaged from Berkeley Earth, whose
[data terms](https://berkeleyearth.org/data/) state **CC BY-NC 4.0** for
general non-commercial use. Check both sources and their current terms before
redistributing or using the data; the Kaggle listing does not replace the
upstream terms.

## Run

From the repository root, run:

```sh
python WeatherProject/src/weather_project.py
```

The analysis uses London data from January 2000 through December 2012. It
requires a complete monthly series in that period and stops with an error if
months are missing. The final 12 months (January–December 2012) are held out
from model selection. Selection uses five expanding-window rolling-origin
splits over the earlier data. Each split forecasts 12 months, and the training
window expands by 12 months between splits; at least 120 months are required
for the selection period. Mean MSE and MAE across those splits are reported
for ARIMA, a seasonal-naive forecast (the previous year's monthly values), and
a SARIMA model with a 12-month seasonal period. The candidate with the lowest
mean MSE is selected (mean MAE breaks ties), then evaluated once on the
untouched 2012 holdout. Holdout metrics are reported separately and do not
affect model selection. After evaluation, the selected model is refit on the
full available series before saving for future use.
ARIMA and SARIMA results are saved in statsmodels format; a seasonal-naive
result is saved as a pickle containing its seasonal period and latest values.

The SQLite database is created at `data/weatherDatabase.db`. Fitted ARIMA
models are saved under `models/`. Each database result records the selected
model type, its metrics and settings, the saved model path, and SHA-256 hashes
of the script and source dataset so an experiment's inputs can be identified. The recorded
training dates refer to the first selection split; test dates span the
selection folds. The holdout columns separately record the selected model's
holdout MSE/MAE and the holdout date range.
The `arima_order_p/d/q` columns always describe the ARIMA candidate; legacy
`order_p/d/q` columns are retained for compatibility and contain the same values.

## License

This project is licensed under the MIT License.
