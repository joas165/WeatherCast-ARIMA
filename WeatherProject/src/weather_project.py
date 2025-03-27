import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Add src/ to sys.path

from connector_class import SQLiteConnector

# Define relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "GlobalLandTemperaturesByMajorCity.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Check if CSV exists
if not os.path.exists(CSV_PATH):
    print(f"Data file not found: {CSV_PATH}. Please add the dataset.")
    exit()

# Load data
df = pd.read_csv(CSV_PATH, parse_dates=['dt'], index_col='dt')

# Show the last few rows of the data
print("Tail of the data:")
print(df.tail(20))

#Choose a major city
city = 'London'

# Load and preprocess data
df = df[(df.index >= '2000-01-01') & (df.index <= '2012-12-01')]  # Restrict data from 2000 to 2010
df = df[df['City'] == city]  # Filter data for the City
df_monthly = df['AverageTemperature'].dropna()

# Show the first few rows of the data
print("Head of the filtered data:")
print(df_monthly.head(20))

# Train-test split
train_size = int(len(df_monthly) * 0.90)
train, test = df_monthly[0:train_size], df_monthly[train_size:]

# Record start and end of train and test sets
train_start = train.index[0].strftime('%Y-%m')
train_end = train.index[-1].strftime('%Y-%m')
test_start = test.index[0].strftime('%Y-%m')
test_end = test.index[-1].strftime('%Y-%m')

# Visualize the train set
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train Set')
plt.title('Monthly Average Temperature - Train Set')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()


# ACF and PACF plots
print("Show ACF and PACF plots")
plt.figure(figsize=(10, 6))
plt.subplot(211)
plot_acf(train, ax=plt.gca())
plt.subplot(212)
plot_pacf(train, ax=plt.gca())
plt.tight_layout()
plt.show()

# ADF test for stationarity
result = adfuller(train)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
adf_stat = result[0]
adf_pvalue = result[1]
adf_critical_values = result[4]

# Ensure stationarity (differencing if needed)
train_diff = train.diff().dropna()

# Re-check stationarity with ADF test after differencing
result_diff = adfuller(train_diff)
print('ADF Statistic (Differenced):', result_diff[0])
print('p-value (Differenced):', result_diff[1])
adf_stat_diff = result_diff[0]
adf_pvalue_diff = result_diff[1]
adf_critical_values_diff = result_diff[4]

# Plot the differenced data
plt.figure(figsize=(10, 6))
plt.plot(train_diff, label='Differenced Temperature')
plt.title('Differenced Monthly Average Temperature')
plt.xlabel('Date')
plt.ylabel('Differenced Temperature')
plt.legend()
plt.show()

# ACF and PACF plots
plt.figure(figsize=(10, 6))
plt.subplot(211)
plot_acf(train_diff, ax=plt.gca())
plt.subplot(212)
plot_pacf(train_diff, ax=plt.gca())
plt.tight_layout()
plt.show()


# Fit ARIMA model (test the parameters)
order = (7, 1, 3)
model = ARIMA(train, order=order)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Visualize residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals')
plt.title('Residuals of ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# Plot residual density
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residual Density Plot')
plt.xlabel('Residuals')
plt.show()

# Forecasting
forecast_steps = len(test)  # Forecast for the length of the test set
forecast = model_fit.forecast(steps=forecast_steps)

# Calculate performance metrics
mse = mean_squared_error(test, forecast)
mae = mean_absolute_error(test, forecast)
print('Test MSE: %.3f' % mse)
print('Test MAE: %.3f' % mae)

# Get 95% confidence intervals
forecast_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()

# Extract lower and upper bounds
forecast_lower = forecast_ci.iloc[:, 0]
forecast_upper = forecast_ci.iloc[:, 1]

# Visualize the predictions vs actuals (only test set)
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Observed')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.fill_between(test.index, forecast_lower, forecast_upper, color='gray', alpha=0.2, label='Uncertainty')
plt.title('Observed vs Forecasted Values (Test Set)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Visualize the errors
errors = test - forecast
plt.figure(figsize=(10, 6))
plt.plot(test.index, errors, label='Errors', color='green')
plt.title('Forecast Errors Over Time')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.show()

# Save parameters and results to SQLite database
print("Saving parameters...")
db_connector = SQLiteConnector()
db_connector.create_table()

db_connector.insert_model_results(
    adf_stat, adf_pvalue, adf_stat, adf_pvalue,
    order, forecast_steps, model_fit.summary().as_text(), mse, mae,
    city, train_start, train_end, test_start, test_end
)

print("Parameters saved")