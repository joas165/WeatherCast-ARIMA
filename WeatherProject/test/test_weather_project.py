import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "src"))
sys.path.append(SRC_DIR)

from weather_project import (
    calculate_metrics,
    forecast_series,
    _fit_arima,
    load_monthly_series,
    rolling_origin_splits,
    seasonal_naive_forecast,
    SeasonalNaiveModel,
    select_best_model,
)


class TestWeatherForecastHelpers(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2000-01-01", periods=24, freq="MS")
        self.series = pd.Series(np.arange(24, dtype=float), index=self.index)

    def test_seasonal_naive_repeats_previous_year_with_test_index(self):
        test_index = pd.date_range("2002-01-01", periods=15, freq="MS")

        forecast = seasonal_naive_forecast(self.series, test_index)

        np.testing.assert_array_equal(forecast.iloc[:12].to_numpy(), self.series.iloc[-12:].to_numpy())
        np.testing.assert_array_equal(forecast.iloc[12:].to_numpy(), self.series.iloc[-12:-9].to_numpy())
        self.assertTrue(forecast.index.equals(test_index))

    def test_seasonal_naive_model_can_be_saved_as_forecaster(self):
        model = SeasonalNaiveModel(self.series.to_numpy())

        saved_model = pickle.loads(pickle.dumps(model.to_dict()))
        restored_model = SeasonalNaiveModel(
            saved_model["last_values"],
            saved_model["seasonal_period"],
        )
        np.testing.assert_array_equal(
            restored_model.forecast(15),
            np.tile(self.series.iloc[-12:].to_numpy(), 2)[:15],
        )

    def test_select_best_model_uses_mean_mse_and_mae_to_break_ties(self):
        metrics = {
            "arima": (2.0, 1.5),
            "sarima": (1.0, 0.8),
            "seasonal_naive": (1.0, 0.7),
        }

        self.assertEqual(select_best_model(metrics), "seasonal_naive")

    def test_arima_fit_uses_higher_iteration_limit_and_rejects_nonconvergence(self):
        with patch("weather_project.ARIMA") as arima:
            arima.return_value.fit.return_value.mle_retvals = {
                "converged": False,
                "iterations": 200,
            }

            with self.assertRaisesRegex(RuntimeError, "failed to converge after 200 iterations"):
                _fit_arima(self.series)

        arima.return_value.fit.assert_called_once_with(
            method_kwargs={"maxiter": 200}
        )

    def test_metrics_reject_misaligned_indexes(self):
        with self.assertRaisesRegex(ValueError, "indexes must match"):
            calculate_metrics(self.series, self.series.iloc[::-1])

    def test_forecast_series_rejects_wrong_number_of_values(self):
        with self.assertRaisesRegex(ValueError, "Expected 24 forecast values, got 2"):
            forecast_series([1, 2], self.index)

    def test_rolling_origin_splits_expand_training_and_cover_each_test_once(self):
        series = pd.Series(
            np.arange(156, dtype=float),
            index=pd.date_range("2000-01-01", periods=156, freq="MS"),
        )

        splits = rolling_origin_splits(series)

        self.assertEqual(len(splits), 5)
        self.assertEqual([len(train) for train, _ in splits], [96, 108, 120, 132, 144])
        self.assertTrue(all(len(test) == 12 for _, test in splits))
        self.assertTrue(all(train.index[-1] < test.index[0] for train, test in splits))
        self.assertTrue(splits[0][1].index[-1] < splits[1][1].index[0])
        self.assertEqual(splits[-1][1].index[-1], series.index[-1])

    def test_rolling_origin_splits_reject_insufficient_history(self):
        with self.assertRaisesRegex(ValueError, "requires at least 120"):
            rolling_origin_splits(self.series)

    def test_loader_sorts_and_returns_month_start_frequency(self):
        frame = pd.DataFrame({
            "dt": self.index[::-1],
            "City": ["London"] * len(self.index),
            "AverageTemperature": np.arange(24, dtype=float)[::-1],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "temperatures.csv")
            frame.to_csv(csv_path, index=False)

            monthly = load_monthly_series(csv_path, "London", "2000-01-01", "2001-12-01")

        self.assertTrue(monthly.index.is_monotonic_increasing)
        self.assertEqual(monthly.index.freqstr, "MS")
        np.testing.assert_array_equal(monthly.to_numpy(), np.arange(24, dtype=float))

    def test_loader_explains_when_dataset_file_is_missing(self):
        missing_path = os.path.join(tempfile.gettempdir(), "missing-temperatures.csv")

        with self.assertRaisesRegex(FileNotFoundError, "root-level data/ directory"):
            load_monthly_series(missing_path, "London", "2000-01-01", "2001-12-01")

    def test_loader_rejects_missing_months(self):
        index = self.index.delete(5)
        frame = pd.DataFrame({
            "dt": index,
            "City": ["London"] * len(index),
            "AverageTemperature": np.arange(len(index), dtype=float),
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "temperatures.csv")
            frame.to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing months: 2000-06"):
                load_monthly_series(csv_path, "London", "2000-01-01", "2001-12-01")


if __name__ == "__main__":
    unittest.main()