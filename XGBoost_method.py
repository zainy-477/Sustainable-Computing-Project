# 1. Imports
import time
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap


## 2. AI / ML Method - Monthly Forecast Function

# 2.1 Monthly data preparation function
def prepare_monthly_features(df_m):

    # 2.1.1 Create an X dataframe to hold features
    X = pd.DataFrame(index=df_m.index)
    y = df_m['CARBON_INTENSITY']
    origin = pd.Timestamp("2009-01-01")
    X['t'] = (df_m.index.year - origin.year) * 12 + (df_m.index.month - origin.month)
    m = X.index.month
    X['sin_month'] = np.sin(2 * np.pi * m / 12)
    X['cos_month'] = np.cos(2 * np.pi * m / 12)

    # 2.1.2 Encode lag features
    X['lag_1'] = y.shift(1)
    X['lag_3'] = y.shift(3)
    X['lag_12'] = y.shift(12)

    # 2.1.3 Encode rolling features
    X['rolling_mean_3'] = y.shift(1).rolling(3).mean()
    X['rolling_mean_12'] = y.shift(1).rolling(12).mean()
    X['roll_std_12'] = y.shift(1).rolling(12).std()

    # 2.1.4 Remove rows with NaN values and return features and target
    X = X.dropna()
    y = y.loc[X.index]
    return X, y

# 2.2 Monthly rolling-origin validation function
def rolling_validation_monthly(df, start_v_year=2018):

    # 2.2.1 Loop through all years from the starting validation year
    df_m = df.resample('ME').mean()
    results = []
    for year in range(start_v_year, 2024):
        print(f"Validating monthly on year: {year+1}")

        # 2.2.2 Prepare training and validation datasets
        train = df_m.loc[:f"{year}-12"]
        val = df_m.loc[f"{year+1}-01":f"{year+1}-12"]

        X_train, y_train = prepare_monthly_features(train)
        feature_base = pd.concat([train.tail(12), val])

        X_val_all, y_val_all = prepare_monthly_features(feature_base)
        X_val = X_val_all.loc[val.index]
        y_val = y_val_all.loc[val.index]

        # 2.2.3 Train XGBoost model and analyse results
        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        best_iteration = model.best_iteration
        print(f"Year {year+1} MAE: {mae:.2f} gCO2/kWh at iteration {best_iteration}")
        results.append([year+1, mae, best_iteration])

        # 2.2.4 Diagnostic Graphs
        ''' 
        fitted = model.predict(X_train)
        plt.plot(y_train.index, y_train, label='Actual')
        plt.plot(y_train.index, fitted, label='Predicted')
        plt.title(f'Carbon Intensity Forecasting for Year {year+1}')
        plt.xlabel('Date')
        plt.ylabel('Carbon Intensity (gCO2/kWh)')
        plt.legend()
        plt.show()
        
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train)
        shap.summary_plot(shap_values, X_train, show=False)
        ax = plt.gca()
        ax.set_xlim(-50, 50)   # choose a range appropriate for gCO2/kWh
        ax.set_xlabel("SHAP value (impact on model output)")
        '''


    return pd.DataFrame(results, columns=['Year', 'MAE', 'Best_Iteration'])

# 2.3 Train monthly xgboost model on all training data compute 2025 MAE
def prediction_monthly(df, iterations):
    
    # 2.3.1 Prepare training and validation datasets
    df_m = df.resample('ME').mean()
    train = df_m.loc[:"2024-12"]
    val = df_m.loc[f"2025-01":f"2025-12"]

    # 2.3.2 Train the model
    X_train, y_train = prepare_monthly_features(train)

    model = xgb.XGBRegressor(
        n_estimators=iterations,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:absoluteerror',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    fitted = model.predict(X_train)

    # 2.3.3 Iteratively predict monthly CI for 2025
    history = train.copy()
    for month in pd.date_range("2025-01-31", "2025-12-31", freq='ME'):
        history.loc[month] = np.nan
        X_all, _ = prepare_monthly_features(history)
        X_month = X_all.loc[[month]]
        val.loc[month, "fitted"] = model.predict(X_month)[0]
        history.loc[month, 'CARBON_INTENSITY'] = val.loc[month, "fitted"]

    # 2.3.4 Plot results and compute MAE
    print(f"Output Accuracy: {mean_absolute_error(val['CARBON_INTENSITY'], val['fitted'])}")

    fig, ax = plt.subplots(2, 1, sharex=False)
    ax[0].plot(y_train.index, y_train, label="Observed", marker="o", color='blue')
    ax[0].plot(y_train.index, fitted, label="Model Fit", linestyle="--", color='red')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(val.index, val['CARBON_INTENSITY'], label="Observed", marker="o", color='blue')
    ax[1].plot(val.index, val['fitted'], label="Model Prediction", linestyle="--", color='red')
    ax[1].legend()
    ax[1].grid(True)

    fig.supxlabel('Year')
    fig.supylabel('Carbon intensity')
    fig.suptitle('Regression Model Performance and Forecast')
    plt.show(block=False)


## 3. AI/ML Method - Hourly Forecast Function

# 3.1 Hourly data preparation function
def prepare_hourly_features(df_h):

    # 3.1.1 Create an X dataframe to hold features
    X = pd.DataFrame(index=df_h.index)
    y = df_h['CARBON_INTENSITY']
    origin = pd.Timestamp("2009-01-01")
    X['t'] = ((df_h.index - origin).total_seconds() / 3600).astype(int)

    # 3.1.2 Encode cyclical featuers
    hour = df_h.index.hour
    dow = df_h.index.dayofweek
    month = df_h.index.month

    X['sin_hour'] = np.sin(2 * np.pi * hour / 24)
    X['cos_hour'] = np.cos(2 * np.pi * hour / 24)

    X['sin_day'] = np.sin(2 * np.pi * dow / 7)
    X['cos_day'] = np.cos(2 * np.pi * dow / 7)

    X['sin_month'] = np.sin(2 * np.pi * month / 12)
    X['cos_month'] = np.cos(2 * np.pi * month / 12)

    # 3.1.3 Encode lag features
    X['lag_1'] = y.shift(1)
    X['lag_24'] = y.shift(24)
    X['lag_168'] = y.shift(168)  

    # 3.1.4 Encode rolling features
    X['rolling_mean_24'] = y.shift(1).rolling(24).mean()
    X['rolling_mean_168'] = y.shift(1).rolling(168).mean()
    X['rolling_std_168'] = y.shift(1).rolling(168).std()

    # 3.1.5 Remove rows with NaN values and return features and target
    X = X.dropna()
    y = y.loc[X.index]
    return X, y

# 3.2 Hourly rolling-origin validation function
def rolling_validation_hourly(df, start_v_year=2018):

    # 3.2.1 Loop through all years from the starting validation year
    df_h = df.resample('H').mean()
    results = []
    for year in range(start_v_year, 2024):
        print(f"Validating hourly on year: {year+1}")

        # 3.2.2 Prepare training and validation datasets
        train = df_h.loc[:f"{year}-12-31 23:00"]
        val = df_h.loc[f"{year+1}-01-01":f"{year+1}-12-31 23:00"]

        X_train, y_train = prepare_hourly_features(train)

        feature_base = pd.concat([train.tail(168), val])
        X_val_all, y_val_all = prepare_hourly_features(feature_base)

        X_val = X_val_all.loc[val.index]
        y_val = y_val_all.loc[val.index]

        # 3.2.3 Train XGBoost model and analyse results
        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:absoluteerror',
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        results.append([year+1, mae, model.best_iteration])

        print(f"Year {year+1} MAE: {mae:.2f}")

    return pd.DataFrame(results, columns=['Year', 'MAE', 'Best_Iteration'])

# 3.3 Train hourly xgboost model on all training data compute 2025 MAE
def prediction_hourly(df, iterations):

    # 3.3.1 Prepare training and validation datasets
    df_h = df.resample('H').mean()
    train = df_h.loc[:'2024-12-31 23:00']
    val = df_h.loc['2025-01-01':'2025-12-31 23:00']

    # 3.3.2 Train the model
    X_train, y_train = prepare_hourly_features(train)

    model = xgb.XGBRegressor(
        n_estimators=iterations,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
   
   # 3.3.3 Iteratively predict hourly CI for 2025
    history = train.copy()
    preds = []

    for ts in val.index:
        print(ts)
        history.loc[ts] = np.nan
        X_all, _ = prepare_hourly_features(history)
        X_ts = X_all.loc[[ts]]
        y_hat = model.predict(X_ts)[0]

        preds.append(y_hat)
        history.loc[ts, 'CARBON_INTENSITY'] = y_hat

    # 3.3.4 Compute MAE
    val['fitted'] = preds
    print(f"Hourly Output Accuracy: {mean_absolute_error(val['CARBON_INTENSITY'], val['fitted'])}")


# 4. Main Execution
if __name__ == "__main__":

    # 5.1 Load dataset
    df = pd.read_csv('df_fuel_ckan.csv', parse_dates=['DATETIME'], usecols=['DATETIME', 'CARBON_INTENSITY'], index_col='DATETIME')

    # 5.2 Monthly Model of XGBoost

    # 5.2.1 Perform rolling-origin validation
    start_time = time.perf_counter()
    results_df = rolling_validation_monthly(df, start_v_year=2018)
    print("\nOverall Validation Results:")
    print(results_df)

    # 5.2.2 Perform final prediction for 2025 (using median of rolling-origin iterations)
    iterations = int(results_df['Best_Iteration'].median())
    prediction_monthly(df, iterations=iterations)
    end_time = time.perf_counter()
    print(f"\n[Monthly] Total execution time: {end_time - start_time:.4f} seconds")

    # 5.3 Hourly Model of XGBoost
    start_time = time.perf_counter()
    results_df = rolling_validation_hourly(df, start_v_year=2018)
    print("\nOverall Hourly Validation Results:")
    print(results_df)
    iterations = int(results_df['Best_Iteration'].median())
    prediction_hourly(df, iterations=iterations)
    end_time = time.perf_counter()
    print(f"\n[Hourly] Total execution time: {end_time - start_time:.4f} seconds")
