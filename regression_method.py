# regression_method.py - tests the performance of the regression method (ML) for carbon intensity
# Zain Hirji, 16th December 2025

# 1.1 Imports
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Ridge

# 1.2 Module-level Caches (system optimisation)
prelinear_cache = {}
powerlaw_cache = {}
sin_cos_cache = {}

# 2. Regression Method - Monthly Forecast Function
def regression_monthly(df_m, v_year=2025, breakpoint=48, tau_s=24.0, ridge_alpha=1.0):
    
    # 2.1 Prepare data for regression 
    dft_m = df_m[df_m.index.year < v_year].copy()
    dfv_m = df_m[df_m.index.year == v_year].copy()
    dft_m['t'] = np.arange(len(dft_m))
    pre = dft_m[dft_m['t'] <= breakpoint].copy()
    post = dft_m[dft_m['t'] >= breakpoint].copy()
    
    # 2.2 Model the pre-breakpoint data
    pre['sin'], pre['cos'] = get_sin_cos(len(pre['t']), 12)

    if breakpoint not in prelinear_cache:
        y_pre = pre['CARBON_INTENSITY']
        X_pre = sm.add_constant(pre[['t', 'sin', 'cos']])
        model_pre = sm.OLS(y_pre, X_pre).fit()
        prelinear_cache[breakpoint] = model_pre

    model_pre = prelinear_cache[breakpoint]
    pre['fitted'] = model_pre.fittedvalues
    a0 = float(pre['fitted'].iloc[-1])     
    
    # 2.3 Model the post-breakpoint data
    post['t_post'] = np.arange(len(post))
    t = post['t_post'].values.astype(float)
    y = post['CARBON_INTENSITY'].values.astype(float)

    # 2.3.1 Annual aspect - power law decay
    def power_law_trend(t, a, tau, p):
        return a * (1+(t/tau))**(-p)
    
    if (breakpoint, v_year) not in powerlaw_cache:
        tau0 = 36.0
        p0 = 1.0
        bounds = ([a0-100, 1e-3, 1e-3], [a0+100, 1e4, 50.0])
        params, _ = curve_fit(power_law_trend, t, y, p0=[a0, tau0, p0], bounds=bounds, maxfev=20000)
        powerlaw_cache[(breakpoint, v_year)] = params
    
    a_hat, tau_hat, p_hat = powerlaw_cache[(breakpoint, v_year)]
    post['annual'] = power_law_trend(t, a_hat, tau_hat, p_hat)
    post['residual'] = post['CARBON_INTENSITY'] - post['annual']

    # 2.3.2 Seasonal aspect - using Ridge to reduce multicollinearity and incorporating a fixed point for continuity
    h = 1 / (1 + t/tau_s)
    sin, cos = get_sin_cos(len(t), 12)

    X_seasonal = sm.add_constant(np.column_stack([sin, cos, h*sin, h*cos]))
    y_seasonal = post['residual'].values
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X_seasonal, y_seasonal)
    post['seasonal'] = ridge.predict(X_seasonal)
    
    post['fitted'] = post['annual'] + post['seasonal']
    pre = pre.drop(pre.index[-1])  
    dft_m['fitted'] = pd.concat([pre['fitted'], post['fitted']]) 

    # 2.4 Apply to predict monthly CI in 2025
    start = post['t_post'].iloc[-1] + 1
    t_p = np.arange(start, start + len(dfv_m))
    h_p = 1 / (1 + t_p/tau_s)
    sin_p, cos_p = get_sin_cos(len(t_p), 12, start)
    X_pred = sm.add_constant(np.column_stack([sin_p, cos_p, h_p*sin_p, h_p*cos_p]))

    dfv_m['annual'] =  power_law_trend(t_p, a_hat, tau_hat, p_hat)
    dfv_m['seasonal'] = ridge.predict(X_pred)
    dfv_m['fitted'] = dfv_m['annual'] + dfv_m['seasonal']

    return dft_m, dfv_m, a_hat, tau_hat, p_hat


# 3. Helper functions

# 3.1 Rolling-origin validation for scoring of hyperparameters (weighted to favour more recent years)
def monthly_score(df_m, b, ts, a):
    score = 0
    for v_year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        _, dfv_m, _, _, _ = regression_monthly(df_m, v_year=v_year, breakpoint=b, tau_s=ts, ridge_alpha=a)
        score += mean_absolute_error(dfv_m['CARBON_INTENSITY'], dfv_m['fitted']) * np.exp(-(2024 - v_year)/4)/4.044
    return score

def hourly_score(df_h, a_hat, tau_hat, p_hat, breakpoint, tau_s, ridge_alpha):
    score = 0
    for v_year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        _, dfv_h = regression_hourly(df_h, a_hat, tau_hat, p_hat, breakpoint, tau_s, ridge_alpha, v_year=v_year)
        score += mean_absolute_error(dfv_h['CARBON_INTENSITY'], dfv_h['fitted']) * np.exp(-(2024 - v_year)/4)/4.044
    return score

# 3.2 Grid search to optimise hyperameters
def find_best(grid, f):
    best_x = None
    best_val = float('inf')
    for x in grid:
        val = f(x)
        if val < best_val:
            best_val = val
            best_x = x
    return best_x, best_val

# 3.3 Sinusoidal cache function
def get_sin_cos(n, period, start=0):
    key = (n, period, start)
    if key not in sin_cos_cache:
        x = np.arange(start, start+n)
        sin_cos_cache[key] = (np.sin(2*np.pi*x/period), np.cos(2*np.pi*x/period))
    return sin_cos_cache[key]
    
    

# 4. Optimisation & graphing for monthly regression
def run_regression_monthly(df, graph=False):
    start_time = time.perf_counter()
    df_m = df.resample('ME').mean()

    # 4.1 Optimise hyperparameters of monthly regression

    # 4.1.1 Set desired hyperparameter values
    breakpoint, tau_s, alpha = 54, 24, 2
    alpha_grid = [0.1, 0.5, 1, 2, 3, 5, 7, 10]
    tau_grid = [6, 12, 18, 24, 30, 36, 48, 60]
    breakpoint_grid = range(48, 61)
    best_mae_a, best_mae_t, best_mae_b = 0, 1, 2
    iterations = 0

    # 4.1.2 Initialise cache for runtime improvement
    prelinear_cache = {}
    powerlaw_cache = {}

    # 4.1.3 Iterate through gradient search until stable score reached
    for it in range(5):
        alpha, best_mae_a = find_best(alpha_grid, lambda a: monthly_score(df_m, breakpoint, tau_s, a))
        iterations += 8
        #print(best_mae_a)
        if best_mae_a == best_mae_t == best_mae_b:
            break
        tau_s, best_mae_t = find_best(tau_grid, lambda ts: monthly_score(df_m, breakpoint, ts, alpha))
        iterations += 8
        #print(best_mae_t)
        if best_mae_a == best_mae_t == best_mae_b:
            break
        breakpoint, best_mae_b = find_best(breakpoint_grid, lambda b: monthly_score(df_m, b, tau_s, alpha))
        iterations += 12
        #print(best_mae_b)
        if best_mae_a == best_mae_t == best_mae_b:
            break

    dft_m, dfv_m, a_hat, tau_hat, p_hat = regression_monthly(df_m, v_year=2025, breakpoint=breakpoint, tau_s=tau_s, ridge_alpha=alpha)
    end_time = time.perf_counter()

    # 4.2 Performance evaluation of our optimised monthly regression 
    print(f"RV-MAE = {best_mae_t:.4f} and Training MAE = {mean_absolute_error(dft_m['CARBON_INTENSITY'], dft_m['fitted']):.4f}")
    print(f"alpha={alpha}, breakpoint={breakpoint}, tau_s={tau_s}, iterations={iterations}")
    print(f"[Regression, Monthly] MAE: {mean_absolute_error(dfv_m['CARBON_INTENSITY'],dfv_m['fitted']):.4f}")
    print(f"[Regression, Monthly] RMSE: {root_mean_squared_error(dfv_m['CARBON_INTENSITY'],dfv_m['fitted']):.4f}")
    print(f"[Monthly] Benchmark MAE (using 2024 values for 2025): {mean_absolute_error(dfv_m['CARBON_INTENSITY'],dft_m[dft_m.index.year==2024]['CARBON_INTENSITY']):.4f}")
    print(f"[Monthly] Benchmark RMSE (using 2024 values for 2025): {root_mean_squared_error(dfv_m['CARBON_INTENSITY'],dft_m[dft_m.index.year==2024]['CARBON_INTENSITY']):.4f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


    # 4.3 Graphs

    # 4.3.1 Plot training data
    plt.figure(figsize=(10, 5))
    plt.plot(dft_m.index, dft_m['CARBON_INTENSITY'], label="Observed", marker="o", color='blue')
    plt.plot(dft_m.index, dft_m['fitted'], label="Model Fit", linestyle="--", color='red')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Year')
    plt.ylabel('Carbon intensity')
    plt.title('Regression Model - Monthly Training Data')
    if graph:
        plt.show()
    else:
        plt.savefig(os.devnull)
        plt.close()
    

    # 4.3.2 Plot validation data
    plt.figure(figsize=(10, 5))
    plt.plot(dfv_m.index, dfv_m['CARBON_INTENSITY'], label="Actual Monthly CI Average", marker="o", color='blue')
    plt.plot(dfv_m.index, dfv_m['fitted'], label="Model Prediction", linestyle="--", color='red')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Carbon intensity')
    plt.title('Regression Model - Monthly Forecast')
    if graph:
        plt.show()
    else:
        plt.savefig(os.devnull)
        plt.close()

    # 4.4 Return variables to be reused in hourly regression
    breakpoint = dft_m.index[breakpoint].to_period('M').to_timestamp(how='start')
    return dft_m, dfv_m, a_hat, tau_hat, p_hat, breakpoint, tau_s

# 5. Regression Method - Hourly Forecast Function
def regression_hourly(df_h, y0, tau_hat, p_hat, breakpoint, tau_s, ridge_alpha, v_year=2025):

    # 5.1 Prepare post-breakpoint data for regression
    df_h = df_h[df_h.index >= breakpoint].copy()
    df_h['t'] = np.arange(len(df_h))
    df_h['hour'] = df_h.index.hour
    df_h['weekend'] = (df_h.index.dayofweek >= 5).astype(int)
    dft_h = df_h[df_h.index.year < v_year].copy()
    dfv_h = df_h[df_h.index.year == v_year].copy()  
    
    # 5.2 Fit pre-defined annual power law decay model
    t = dft_h['t'].values.astype(float)
    tau_hat = tau_hat * 8760/12
    dft_h['annual'] = y0 * (1+(t/tau_hat))**(-p_hat)
    dft_h['residual'] = dft_h['CARBON_INTENSITY'] - dft_h['annual']

    # 5.3 Ridge regression for seasonal & weekly/hourly patterns

    # 5.3.1 Seasonal features
    tau_s = tau_s * 30 * 24
    h_s = 1 / (1 + t/tau_s)
    sin, cos = get_sin_cos(len(t), 8760)

    # 5.3.2 Weekly/hourly features
    h = dft_h['hour'].values.astype(float)
    weekend = dft_h['weekend'].values.astype(float)
    sina, cosa = get_sin_cos(len(h), 24) 
    sinb, cosb = get_sin_cos(len(h), 12)

    # 5.3.3 Ridge regression    
    X = np.column_stack([sin, cos, h_s*sin, h_s*cos, sina, cosa, sinb, cosb, weekend, weekend*sina, weekend*cosa, weekend*sinb, weekend*cosb])
    X = sm.add_constant(X)
    y =  dft_h['residual']
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X, y)
    dft_h['regression'] = ridge.predict(X)
    dft_h['fitted'] = dft_h['annual'] + dft_h['regression']
    
    # 5.4 Apply to predict hourly CI in 2025
    start = dft_h['t'].iloc[-1] + 1
    t_p = np.arange(start, start + len(dfv_h))
    h_sp = 1 / (1 + t_p/tau_s)
    sin_p, cos_p = get_sin_cos(len(t_p), 8760, start)

    h_p = dfv_h['hour'].values.astype(float)
    weekend_p = dfv_h['weekend'].values.astype(float)
    sina_p, cosa_p = get_sin_cos(len(h_p), 24)
    sinb_p, cosb_p = get_sin_cos(len(h_p), 12)
    X_p = np.column_stack([sin_p, cos_p, h_sp*sin_p, h_sp*cos_p, sina_p, cosa_p, sinb_p, cosb_p, weekend_p, weekend_p*sina_p, weekend_p*cosa_p, weekend_p*sinb_p, weekend_p*cosb_p])
    X_p = sm.add_constant(X_p)

    dfv_h['annual'] = y0 * (1+(t_p/tau_hat))**(-p_hat)
    dfv_h['regression'] = ridge.predict(X_p)
    dfv_h['fitted'] = dfv_h['annual'] + dfv_h['regression']

    return dft_h, dfv_h

# 6. Optimisation and graphing for hourly regression
def run_regression_hourly(df, a_hat, tau_hat, p_hat, breakpoint, tau_s, graph=False):
    start_time = time.perf_counter()
    df_h = df.resample('h').mean()

    # 6.1 Optimise our alpha hyperparameter
    alpha_grid = [50, 100, 150, 200, 250, 300, 350, 400]
    best_alpha, _ = find_best(alpha_grid, lambda a: hourly_score(df_h, a_hat, tau_hat, p_hat, breakpoint, tau_s, a))
    dft_h, dfv_h = regression_hourly(df_h, a_hat, tau_hat, p_hat, breakpoint, tau_s, best_alpha)

    # 6.2 Performance evaluation of our optimised hourly regression
    print(f"alpha={best_alpha}")
    print(f"[Regression, Hourly] MAE: {mean_absolute_error(dfv_h['CARBON_INTENSITY'], dfv_h['fitted'])}")
    print(f"[Regression, Hourly] RMSE: {root_mean_squared_error(dfv_h['CARBON_INTENSITY'], dfv_h['fitted'])}")
    ci_2024 = dft_h[(dft_h.index.year==2024) & (dft_h.index.strftime('%m-%d') != '02-29')]['CARBON_INTENSITY']
    print(f"Benchmark MAE (using 2024 values for 2025): {mean_absolute_error(dfv_h['CARBON_INTENSITY'], ci_2024)}")
    print(f"Benchmark RMSE (using 2024 values for 2025): {root_mean_squared_error(dfv_h['CARBON_INTENSITY'], ci_2024)}")
    end_time = time.perf_counter()
    print(f"Execution time (hourly): {end_time - start_time:.4f} seconds")

    # 6.3 Graphs
    plt.figure(figsize=(10, 5))
    date = dfv_h.loc['2025-09-01 00:00:00':'2025-09-14 23:00:00'].copy()
    plt.plot(date.index, date['CARBON_INTENSITY'], label="Observed", marker="o", color='blue')
    plt.plot(date.index, date['fitted'], label="Model Fit", linestyle="--", color='red')
    plt.plot(date.index, date['annual'], label="Annual Trend", linestyle="--", color='green')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel('Carbon intensity')
    plt.title('Regression Model Fit on Data')
    if graph:
        plt.show()
    else:
        plt.savefig(os.devnull)
        plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(dft_h.index, dft_h['CARBON_INTENSITY'], label='Observed', color='blue')
    plt.plot(dft_h.index, dft_h['fitted'], label='Model fit', color='red')
    plt.title('Regression Model - Hourly Training Data')
    plt.xlabel('Year')
    plt.ylabel('Carbon intensity')
    plt.grid(True)
    if graph:
        plt.show()
    else:
        plt.savefig(os.devnull)
        plt.close()

    # Return forecast for plotting
    return dft_h, dfv_h

    
# 7. Regression Model Execution
def regression(show_graphs=False):
    df = pd.read_csv('df_fuel_ckan.csv', parse_dates=['DATETIME'], usecols=['DATETIME', 'CARBON_INTENSITY'], index_col='DATETIME')
    _, _, a_hat, tau_hat, p_hat, breakpoint, tau_s = run_regression_monthly(df, graph=show_graphs)
    run_regression_hourly(df, a_hat, tau_hat, p_hat, breakpoint, tau_s, graph=show_graphs)


# 8. Main execution
if __name__ == "__main__":
    regression(show_graphs=True)