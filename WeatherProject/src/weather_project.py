import hashlib
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from connector_class import SQLiteConnector

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "GlobalLandTemperaturesByMajorCity.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

CITY = "London"
START_DATE = "2000-01-01"
END_DATE = "2012-12-01"
ARIMA_ORDER = (7, 1, 3)
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
EVALUATION_FOLDS = 5
FORECAST_HORIZON_MONTHS = 12
MIN_TRAIN_MONTHS = 60


def load_monthly_series(csv_path, city, start_date, end_date):
    """Load a complete, chronologically ordered monthly temperature series."""
    df = pd.read_csv(csv_path, parse_dates=["dt"], index_col="dt")
    required_columns = {"City", "AverageTemperature"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    city_data = df.loc[
        (df.index >= start_date) & (df.index <= end_date) & (df["City"] == city),
        "AverageTemperature",
    ]
    monthly = city_data.sort_index().resample("MS").mean().asfreq("MS")
    if monthly.empty:
        raise ValueError(f"No temperature data found for {city} between {start_date} and {end_date}.")
    if monthly.isna().any():
        missing_months = monthly.index[monthly.isna()].strftime("%Y-%m").tolist()
        raise ValueError(f"Temperature data is missing months: {', '.join(missing_months)}")
    if not np.isfinite(monthly.to_numpy()).all():
        raise ValueError("Temperature data contains non-finite values.")
    return monthly


def forecast_series(values, index):
    """Associate forecast values with the exact observed monthly index."""
    values = np.asarray(values).reshape(-1)
    if len(values) != len(index):
        raise ValueError(f"Expected {len(index)} forecast values, got {len(values)}.")
    return pd.Series(values, index=index)


def calculate_metrics(actual, predicted):
    if not actual.index.equals(predicted.index):
        raise ValueError("Actual and forecast indexes must match exactly.")
    return (
        mean_squared_error(actual, predicted),
        mean_absolute_error(actual, predicted),
    )


def seasonal_naive_forecast(train, index):
    if len(train) < 12:
        raise ValueError("At least 12 training months are required for the seasonal-naive forecast.")
    values = np.tile(train.iloc[-12:].to_numpy(), (len(index) + 11) // 12)[: len(index)]
    return forecast_series(values, index)


def rolling_origin_splits(
    series,
    n_splits=EVALUATION_FOLDS,
    horizon=FORECAST_HORIZON_MONTHS,
    step=FORECAST_HORIZON_MONTHS,
    min_train_size=MIN_TRAIN_MONTHS,
):
    if n_splits < 1 or horizon < 1 or step < 1 or min_train_size < 1:
        raise ValueError("Split count, horizon, step, and minimum training size must be positive.")
    initial_train_size = len(series) - (n_splits - 1) * step - horizon
    if initial_train_size < min_train_size:
        required = min_train_size + (n_splits - 1) * step + horizon
        raise ValueError(
            f"Rolling-origin evaluation requires at least {required} monthly observations; "
            f"got {len(series)}."
        )

    return [
        (
            series.iloc[: initial_train_size + fold * step],
            series.iloc[
                initial_train_size + fold * step:
                initial_train_size + fold * step + horizon
            ],
        )
        for fold in range(n_splits)
    ]


def evaluate_rolling_origin(series):
    folds = rolling_origin_splits(series)
    fold_metrics = {"arima": [], "sarima": [], "seasonal_naive": []}
    last_fold = None

    for train, test in folds:
        arima_fit = ARIMA(train, order=ARIMA_ORDER).fit()
        sarima_fit = SARIMAX(
            train,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        arima_forecast = forecast_series(arima_fit.forecast(steps=len(test)), test.index)
        sarima_forecast = forecast_series(sarima_fit.forecast(steps=len(test)), test.index)
        baseline_forecast = seasonal_naive_forecast(train, test.index)
        fold_metrics["arima"].append(calculate_metrics(test, arima_forecast))
        fold_metrics["sarima"].append(calculate_metrics(test, sarima_forecast))
        fold_metrics["seasonal_naive"].append(calculate_metrics(test, baseline_forecast))

        forecast_result = arima_fit.get_forecast(steps=len(test))
        confidence_interval = forecast_result.conf_int()
        last_fold = {
            "train": train,
            "test": test,
            "model_fit": arima_fit,
            "forecast": arima_forecast,
            "lower": forecast_series(confidence_interval.iloc[:, 0], test.index),
            "upper": forecast_series(confidence_interval.iloc[:, 1], test.index),
        }

    averages = {
        name: tuple(np.mean(metrics, axis=0))
        for name, metrics in fold_metrics.items()
    }
    return averages, last_fold


def _show_diagnostics(train, train_diff, model_fit, test, forecast, lower, upper):
    plt.figure(figsize=(10, 6))
    plt.plot(train, label="Train Set")
    plt.title("Monthly Average Temperature - Train Set")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

    for series, title in ((train, "Training Data"), (train_diff, "Differenced Training Data")):
        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        plot_acf(series, ax=plt.gca())
        plt.subplot(212)
        plot_pacf(series, ax=plt.gca())
        plt.suptitle(f"{title}: ACF and PACF")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(model_fit.resid, label="Residuals")
    plt.title("Residuals of ARIMA Model")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(model_fit.resid, kde=True)
    plt.title("Residual Density Plot")
    plt.xlabel("Residuals")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label="Observed")
    plt.plot(test.index, forecast, label="ARIMA Forecast", color="red")
    plt.fill_between(test.index, lower, upper, color="gray", alpha=0.2, label="95% interval")
    plt.title("Observed vs Forecasted Values (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()


def run_analysis(csv_path=CSV_PATH, db_path=None, show_plots=True):
    monthly = load_monthly_series(csv_path, CITY, START_DATE, END_DATE)
    folds = rolling_origin_splits(monthly)
    initial_train = folds[0][0]
    result = adfuller(initial_train)
    adf_stat, adf_pvalue = result[0], result[1]
    train_diff = initial_train.diff().dropna()
    result_diff = adfuller(train_diff)
    adf_stat_diff, adf_pvalue_diff = result_diff[0], result_diff[1]

    average_metrics, last_fold = evaluate_rolling_origin(monthly)
    arima_mse, arima_mae = average_metrics["arima"]
    sarima_mse, sarima_mae = average_metrics["sarima"]
    baseline_mse, baseline_mae = average_metrics["seasonal_naive"]
    print(
        f"ARIMA{ARIMA_ORDER} rolling-origin mean MSE: {arima_mse:.3f}; "
        f"MAE: {arima_mae:.3f}"
    )
    print(
        f"SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL_ORDER} rolling-origin mean MSE: "
        f"{sarima_mse:.3f}; MAE: {sarima_mae:.3f}"
    )
    print(
        f"Seasonal-naive rolling-origin mean MSE: {baseline_mse:.3f}; "
        f"MAE: {baseline_mae:.3f}"
    )

    if show_plots:
        _show_diagnostics(
            initial_train,
            train_diff,
            last_fold["model_fit"],
            last_fold["test"],
            last_fold["forecast"],
            last_fold["lower"],
            last_fold["upper"],
        )

    model_fit = ARIMA(monthly, order=ARIMA_ORDER).fit()
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_name = f"arima_{CITY.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.pkl"
    model_path = os.path.join(MODEL_DIR, model_name)
    model_fit.save(model_path)
    with open(__file__, "rb") as source_file:
        code_hash = hashlib.sha256(source_file.read()).hexdigest()
    with open(csv_path, "rb") as data_file:
        data_hash = hashlib.sha256(data_file.read()).hexdigest()

    connector = SQLiteConnector(db_path) if db_path else SQLiteConnector()
    connector.create_table()
    connector.insert_model_results(
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        adf_stat_diff=adf_stat_diff,
        adf_pvalue_diff=adf_pvalue_diff,
        order=ARIMA_ORDER,
        forecast_steps=FORECAST_HORIZON_MONTHS,
        model_summary=model_fit.summary().as_text(),
        mse=arima_mse,
        mae=arima_mae,
        city=CITY,
        train_start=initial_train.index[0].strftime("%Y-%m"),
        train_end=initial_train.index[-1].strftime("%Y-%m"),
        test_start=folds[0][1].index[0].strftime("%Y-%m"),
        test_end=folds[-1][1].index[-1].strftime("%Y-%m"),
        seasonal_naive_mse=baseline_mse,
        seasonal_naive_mae=baseline_mae,
        sarima_mse=sarima_mse,
        sarima_mae=sarima_mae,
        sarima_order=SARIMA_ORDER,
        sarima_seasonal_order=SARIMA_SEASONAL_ORDER,
        model_path=os.path.relpath(model_path, PROJECT_DIR),
        code_version=code_hash,
        data_hash=data_hash,
        evaluation_folds=EVALUATION_FOLDS,
        forecast_horizon_months=FORECAST_HORIZON_MONTHS,
    )
    print(f"ARIMA model saved to {model_path}")
    print("Experiment results saved to SQLite.")


if __name__ == "__main__":
    run_analysis()
