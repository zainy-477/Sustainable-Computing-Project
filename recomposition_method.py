# recomposition_method.py - tests the performance of the recomposition method (non-ML) for carbon intensity
# Zain Hirji, 26th December 2025

# 1. Imports
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# 2. Recomposition Method - Monthly Forecast Function
def recomposition_monthly(df, graph=False):
    start_time = time.perf_counter()

    # 2.1 Evaluate the yearly trend using rolling mean
    df_m = df['CARBON_INTENSITY'].resample('ME').mean().copy()
    monthly = df_m[df_m.index.year < 2025]
    annual_trend = monthly.rolling(window=12, center=False, min_periods=12).mean()
    slope = annual_trend.loc['2023-01-31':'2024-12-31'].diff().mean()

    trend_forecast = pd.Series(index=pd.date_range(start='2025-01-31', end='2025-12-31', freq='ME'))
    trend_forecast[:] = annual_trend['2024-12-31'] + slope * np.arange(1, 13)
    
    # 2.2 Evaluate seasonal profile
    seasonal_ratio = monthly / annual_trend
    seasonal_profile = seasonal_ratio.groupby(seasonal_ratio.index.month).mean()

    # 2.3 Generate forecast for 2025
    months = trend_forecast.index.month
    forecast = trend_forecast.values * seasonal_profile.loc[months].values
    forecast = pd.Series(forecast, index=trend_forecast.index)

    # 2.4 Evaluate accuracy of forecast
    actual_2025 = df_m[df_m.index.year == 2025]
    print(f"[Recomposition, Monthly] MAE: {mean_absolute_error(actual_2025, forecast)}")
    print(f"[Recomposition, Monthly] RMSE: {root_mean_squared_error(actual_2025, forecast)}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # 2.5 Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(forecast.index, forecast.values, label='Model Prediction', marker='o')
    plt.plot(actual_2025.index, actual_2025.values, label='Actual Monthly CI Average', marker='o')
    plt.title('Recomposition Method - Monthly Forecast')
    plt.xlabel('Date')
    plt.ylabel('Carbon Intensity')
    plt.legend()
    plt.grid()
    if graph:
        plt.show()
    else:
        plt.savefig(os.devnull)
        plt.close()

    return forecast

# 3. Recomposition Method - Hourly Forecast Function
def recomposition_hourly(df, monthly_forecast, graph=False):
    start_time = time.perf_counter()

    # 3.1 Evaluate the hourly ratio
    df_h = df.resample('h').mean()
    dft_h = df_h[df_h.index.year == 2024]
    dfv_h = df_h[df_h.index.year == 2025]

    monthly_hist = (dft_h.groupby(dft_h.index.to_period('M'))['CARBON_INTENSITY'].mean())
    monthly_actual = (dfv_h.groupby(dfv_h.index.to_period('M'))['CARBON_INTENSITY'].mean())

    hourly = dft_h['CARBON_INTENSITY'].resample('h').mean()
    hourly_month = hourly.index.to_period('M').map(monthly_hist)
    hourly_ratio = hourly / hourly_month
    
    # 3.2 Evaluate seasonal profile
    hour_of_week = hourly.index.hour + 24 * hourly.index.dayofweek
    hourly_profile = hourly_ratio.groupby(hour_of_week).mean()

    # 3.3 Generate forecast for 2025
    monthly_forecast.index = monthly_forecast.index.to_period('M')
    forecast_index = pd.date_range(start='2025-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h')
    how = forecast_index.hour + 24 * forecast_index.dayofweek
    months = forecast_index.to_period('M')

    forecast = hourly_profile.loc[how].values * monthly_forecast.loc[months].values
    hourly_forecast = hourly_profile.loc[how].values * monthly_actual.loc[months].values

    forecast = pd.Series(forecast, index=forecast_index)
    hourly_forecast = pd.Series(hourly_forecast, index=forecast_index)

    # 3.4 Evaluate accuracy of forecast
    actual_2025 = df['CARBON_INTENSITY'].resample('h').mean().loc['2025-01-01 00:00:00':'2025-12-31 23:00:00']
    print(f"[Recomposition, Hourly] MAE (using monthly forecast): {mean_absolute_error(actual_2025, forecast)}")
    print(f"[Recomposition, Hourly] RMSE (using monthly forecast): {root_mean_squared_error(actual_2025, forecast)}")
    print(f"[Recomposition, Hourly] MAE (using monthly values): {mean_absolute_error(actual_2025, hourly_forecast)}")
    print(f"[Recomposition, Hourly] RMSE (using monthly values): {root_mean_squared_error(actual_2025, hourly_forecast)}")

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # 3.5 Plot example of results
    plt.figure(figsize=(10, 5))
    date = forecast.loc['2025-09-01 00:00:00':'2025-09-14 23:00:00'].copy()
    actual = actual_2025.loc['2025-09-01 00:00:00':'2025-09-14 23:00:00'].copy()
    plt.plot(date.index, date.values, label="Model Fit", linestyle="--", color='#172F0A')
    plt.plot(actual.index, actual.values, label="Observed", color='blue')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Carbon intensity')
    plt.title('Recomposition Model Fit on Data')
    if graph:
        plt.show()
    else:
        plt.savefig(os.devnull)
        plt.close()

    return forecast


# 4. Recomposition Model Execution
def recomposition(show_graphs=False):
    df = pd.read_csv('df_fuel_ckan.csv', parse_dates=['DATETIME'], usecols=['DATETIME', 'CARBON_INTENSITY'], index_col='DATETIME')
    monthly_forecast = recomposition_monthly(df, graph=show_graphs)
    recomposition_hourly(df, monthly_forecast, graph=show_graphs)


# 5. Main execution
if __name__ == "__main__":
    recomposition(show_graphs=True)