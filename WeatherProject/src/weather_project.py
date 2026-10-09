import hashlib
import os
import pickle
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
ARIMA_MAXITER = 200
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
EVALUATION_FOLDS = 5
FORECAST_HORIZON_MONTHS = 12
HOLDOUT_HORIZON_MONTHS = 12
MIN_TRAIN_MONTHS = 60


def load_monthly_series(csv_path, city, start_date, end_date):
    """Load a complete, chronologically ordered monthly temperature series."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Download "
            "'GlobalLandTemperaturesByMajorCity.csv' and place it in the "
            "repository's root-level data/ directory; see Readme.md."
        )
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


class SeasonalNaiveModel:
    def __init__(self, values, seasonal_period=12):
        if seasonal_period < 1 or len(values) < seasonal_period:
            raise ValueError("The training series must contain at least one full seasonal period.")
        self.last_values = np.asarray(values[-seasonal_period:], dtype=float)
        self.seasonal_period = seasonal_period

    def forecast(self, steps):
        if steps < 0:
            raise ValueError("Forecast steps cannot be negative.")
        repeats = (steps + self.seasonal_period - 1) // self.seasonal_period
        return np.tile(self.last_values, repeats)[:steps]

    def to_dict(self):
        return {
            "model": "seasonal_naive",
            "seasonal_period": self.seasonal_period,
            "last_values": self.last_values.tolist(),
        }


def seasonal_naive_forecast(train, index):
    if len(train) < 12:
        raise ValueError("At least 12 training months are required for the seasonal-naive forecast.")
    values = SeasonalNaiveModel(train.to_numpy()).forecast(len(index))
    return forecast_series(values, index)


def select_best_model(average_metrics):
    """Select the candidate with the lowest mean MSE, using MAE to break ties."""
    return min(
        average_metrics,
        key=lambda name: (average_metrics[name][0], average_metrics[name][1]),
    )


def _fit_arima(train):
    fit = ARIMA(train, order=ARIMA_ORDER).fit(
        method_kwargs={"maxiter": ARIMA_MAXITER}
    )
    if not fit.mle_retvals.get("converged", False):
        iterations = fit.mle_retvals.get("iterations", "unknown")
        raise RuntimeError(
            f"ARIMA{ARIMA_ORDER} failed to converge after {iterations} iterations "
            f"(maximum {ARIMA_MAXITER})."
        )
    return fit


def _fit_sarima(train):
    return SARIMAX(
        train,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


def split_final_holdout(series, horizon=HOLDOUT_HORIZON_MONTHS):
    if horizon < 1 or len(series) <= horizon:
        raise ValueError("The series must contain more observations than the holdout horizon.")
    return series.iloc[:-horizon], series.iloc[-horizon:]


def evaluate_holdout(train, test, model_name):
    confidence_intervals = None
    if model_name == "arima":
        model_fit = _fit_arima(train)
        forecast_result = model_fit.get_forecast(steps=len(test))
        confidence_intervals = forecast_result.conf_int()
        forecast_values = forecast_result.predicted_mean
    elif model_name == "sarima":
        model_fit = _fit_sarima(train)
        forecast_result = model_fit.get_forecast(steps=len(test))
        confidence_intervals = forecast_result.conf_int()
        forecast_values = forecast_result.predicted_mean
    elif model_name == "seasonal_naive":
        model_fit = None
        forecast_values = seasonal_naive_forecast(train, test.index)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    forecast = forecast_series(forecast_values, test.index)
    return calculate_metrics(test, forecast), forecast, model_fit, confidence_intervals


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
        arima_fit = _fit_arima(train)
        sarima_fit = _fit_sarima(train)

        arima_result = arima_fit.get_forecast(steps=len(test))
        sarima_result = sarima_fit.get_forecast(steps=len(test))
        arima_forecast = forecast_series(arima_result.predicted_mean, test.index)
        sarima_forecast = forecast_series(sarima_result.predicted_mean, test.index)
        baseline_forecast = seasonal_naive_forecast(train, test.index)
        fold_metrics["arima"].append(calculate_metrics(test, arima_forecast))
        fold_metrics["sarima"].append(calculate_metrics(test, sarima_forecast))
        fold_metrics["seasonal_naive"].append(calculate_metrics(test, baseline_forecast))

        last_fold = {
            "train": train,
            "test": test,
            "model_fits": {"arima": arima_fit, "sarima": sarima_fit},
            "forecasts": {
                "arima": arima_forecast,
                "sarima": sarima_forecast,
                "seasonal_naive": baseline_forecast,
            },
            "confidence_intervals": {
                "arima": arima_result.conf_int(),
                "sarima": sarima_result.conf_int(),
            },
        }

    averages = {
        name: tuple(np.mean(metrics, axis=0))
        for name, metrics in fold_metrics.items()
    }
    return averages, last_fold


def _show_diagnostics(train, train_diff, residuals, model_name, test, forecast, lower, upper):
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
    plt.plot(residuals, label="Residuals")
    plt.title(f"Residuals of {model_name} Model")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Density Plot")
    plt.xlabel("Residuals")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label="Observed")
    plt.plot(test.index, forecast, label=f"{model_name} Forecast", color="red")
    if lower is not None and upper is not None:
        plt.fill_between(test.index, lower, upper, color="gray", alpha=0.2, label="95% interval")
    plt.title("Observed vs Forecasted Values (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()


def run_analysis(csv_path=CSV_PATH, db_path=None, show_plots=True):
    monthly = load_monthly_series(csv_path, CITY, START_DATE, END_DATE)
    selection_series, holdout = split_final_holdout(monthly)
    folds = rolling_origin_splits(selection_series)
    initial_train = folds[0][0]
    result = adfuller(initial_train)
    adf_stat, adf_pvalue = result[0], result[1]
    train_diff = initial_train.diff().dropna()
    result_diff = adfuller(train_diff)
    adf_stat_diff, adf_pvalue_diff = result_diff[0], result_diff[1]

    average_metrics, _ = evaluate_rolling_origin(selection_series)
    selected_model = select_best_model(average_metrics)
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
    selected_mse, selected_mae = average_metrics[selected_model]
    model_labels = {
        "arima": "ARIMA",
        "sarima": "SARIMA",
        "seasonal_naive": "Seasonal-naive",
    }
    selected_label = model_labels[selected_model]
    print(
        f"Selected {selected_label} based on the lowest rolling-origin mean MSE "
        f"({selected_mse:.3f}; MAE: {selected_mae:.3f})."
    )
    (holdout_mse, holdout_mae), holdout_forecast, holdout_fit, holdout_intervals = (
        evaluate_holdout(selection_series, holdout, selected_model)
    )
    print(
        f"{selected_label} untouched holdout ({holdout.index[0]:%Y-%m} to "
        f"{holdout.index[-1]:%Y-%m}) MSE: {holdout_mse:.3f}; "
        f"MAE: {holdout_mae:.3f}"
    )

    if show_plots:
        diagnostic_train = selection_series
        if selected_model == "seasonal_naive":
            residuals = diagnostic_train.diff(12).dropna()
        else:
            residuals = holdout_fit.resid
        lower = (
            forecast_series(holdout_intervals.iloc[:, 0], holdout.index)
            if holdout_intervals is not None
            else None
        )
        upper = (
            forecast_series(holdout_intervals.iloc[:, 1], holdout.index)
            if holdout_intervals is not None
            else None
        )
        _show_diagnostics(
            diagnostic_train,
            diagnostic_train.diff().dropna(),
            residuals,
            selected_label,
            holdout,
            holdout_forecast,
            lower,
            upper,
        )

    if selected_model == "arima":
        model_fit = _fit_arima(monthly)
        model_summary = model_fit.summary().as_text()
    elif selected_model == "sarima":
        model_fit = _fit_sarima(monthly)
        model_summary = model_fit.summary().as_text()
    else:
        model_fit = SeasonalNaiveModel(monthly.to_numpy())
        model_summary = (
            f"Seasonal-naive model with period {model_fit.seasonal_period}; "
            f"last seasonal values: {model_fit.last_values.tolist()}"
        )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_name = (
        f"{selected_model}_{CITY.lower()}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.pkl"
    )
    model_path = os.path.join(MODEL_DIR, model_name)
    if selected_model == "seasonal_naive":
        with open(model_path, "wb") as model_file:
            pickle.dump(model_fit.to_dict(), model_file)
    else:
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
        model_summary=model_summary,
        mse=selected_mse,
        mae=selected_mae,
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
        selected_model=selected_model,
        holdout_mse=holdout_mse,
        holdout_mae=holdout_mae,
        holdout_start=holdout.index[0].strftime("%Y-%m"),
        holdout_end=holdout.index[-1].strftime("%Y-%m"),
        holdout_horizon_months=len(holdout),
    )
    print(f"{selected_label} model saved to {model_path}")
    print("Experiment results saved to SQLite.")


if __name__ == "__main__":
    run_analysis()
