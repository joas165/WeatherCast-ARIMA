import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "src"))
sys.path.append(SRC_DIR)

from weather_project import (
    calculate_metrics,
    forecast_series,
    load_monthly_series,
    rolling_origin_splits,
    seasonal_naive_forecast,
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