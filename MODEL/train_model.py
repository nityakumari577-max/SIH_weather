import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# =====================================================
# 1. LOAD DATA
# =====================================================

file_path = "data/DailyDelhiClimateTrain.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# =====================================================
# 2. DATA CLEANING
# =====================================================

df["date"] = pd.to_datetime(df["date"])

# Sort according to date
df = df.sort_values("date")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()


# =====================================================
# 3. REMOVE INVALID PRESSURE VALUES
# =====================================================

# Normal atmospheric pressure is roughly around 1000 hPa.
# Your dataset contains some abnormal values.

df.loc[
    (df["meanpressure"] < 900) |
    (df["meanpressure"] > 1100),
    "meanpressure"
] = np.nan

# Fill abnormal pressure values
df["meanpressure"] = df["meanpressure"].interpolate()

df = df.dropna()


# =====================================================
# 4. DATE FEATURES
# =====================================================

df["day"] = df["date"].dt.day

df["month"] = df["date"].dt.month

df["year"] = df["date"].dt.year

df["day_of_year"] = df["date"].dt.dayofyear


# =====================================================
# 5. LAG FEATURES
# =====================================================

# Previous day's weather

df["temp_lag1"] = df["meantemp"].shift(1)
df["temp_lag2"] = df["meantemp"].shift(2)
df["temp_lag3"] = df["meantemp"].shift(3)

df["humidity_lag1"] = df["humidity"].shift(1)
df["humidity_lag2"] = df["humidity"].shift(2)

df["wind_lag1"] = df["wind_speed"].shift(1)
df["wind_lag2"] = df["wind_speed"].shift(2)

df["pressure_lag1"] = df["meanpressure"].shift(1)
df["pressure_lag2"] = df["meanpressure"].shift(2)


# =====================================================
# 6. TARGET
# =====================================================

# Predict tomorrow's mean temperature

df["target_temp"] = df["meantemp"].shift(-1)


# Remove rows created by shift()
df = df.dropna()


# =====================================================
# 7. FEATURES
# =====================================================

features = [

    "meantemp",
    "humidity",
    "wind_speed",
    "meanpressure",

    "day",
    "month",
    "year",
    "day_of_year",

    "temp_lag1",
    "temp_lag2",
    "temp_lag3",

    "humidity_lag1",
    "humidity_lag2",

    "wind_lag1",
    "wind_lag2",

    "pressure_lag1",
    "pressure_lag2"
]


X = df[features]

y = df["target_temp"]


# =====================================================
# 8. TIME BASED TRAIN TEST SPLIT
# =====================================================

# Weather data is time-series data.
# So DON'T randomly shuffle it.

split = int(len(df) * 0.80)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]


print("\nTraining records:", len(X_train))

print("Testing records:", len(X_test))


# =====================================================
# 9. MODEL 1 - RANDOM FOREST
# =====================================================

rf_model = RandomForestRegressor(

    n_estimators=300,

    max_depth=15,

    random_state=42,

    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_prediction = rf_model.predict(X_test)


# =====================================================
# 10. MODEL 2 - GRADIENT BOOSTING
# =====================================================

gb_model = GradientBoostingRegressor(

    n_estimators=200,

    learning_rate=0.05,

    max_depth=3,

    random_state=42
)

gb_model.fit(X_train, y_train)

gb_prediction = gb_model.predict(X_test)


# =====================================================
# 11. MODEL EVALUATION
# =====================================================

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

gb_mae = mean_absolute_error(
    y_test,
    gb_prediction
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_prediction
    )
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_prediction
    )
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nRandom Forest")

print("MAE :", rf_mae)

print("RMSE:", rf_rmse)


print("\nGradient Boosting")

print("MAE :", gb_mae)

print("RMSE:", gb_rmse)


# =====================================================
# 12. DYNAMIC BLENDING WEIGHTS
# =====================================================

# Better model = lower MAE
# Lower MAE gets higher weight

rf_score = 1 / rf_mae

gb_score = 1 / gb_mae

total_score = rf_score + gb_score


rf_weight = rf_score / total_score

gb_weight = gb_score / total_score


print("\n==============================")
print("BLENDING WEIGHTS")
print("==============================")

print(
    "Random Forest:",
    round(rf_weight, 4)
)

print(
    "Gradient Boosting:",
    round(gb_weight, 4)
)


# =====================================================
# 13. FINAL BLENDED PREDICTION
# =====================================================

blended_prediction = (

    rf_weight * rf_prediction

    +

    gb_weight * gb_prediction
)


# =====================================================
# 14. BLENDED MODEL PERFORMANCE
# =====================================================

blend_mae = mean_absolute_error(

    y_test,

    blended_prediction
)


blend_rmse = np.sqrt(

    mean_squared_error(

        y_test,

        blended_prediction
    )
)


print("\n==============================")
print("BLENDED MODEL")
print("==============================")

print(
    "Blended MAE:",
    blend_mae
)

print(
    "Blended RMSE:",
    blend_rmse
)


# =====================================================
# 15. SAVE PREDICTIONS
# =====================================================

results = pd.DataFrame({

    "date":
        df.iloc[split:]["date"].values,

    "actual_temperature":
        y_test.values,

    "random_forest":
        rf_prediction,

    "gradient_boosting":
        gb_prediction,

    "blended_prediction":
        blended_prediction
})


results.to_csv(

    "weather_predictions.csv",

    index=False
)


# =====================================================
# 16. SAVE ML MODEL
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)


model_package = {

    "random_forest": rf_model,

    "gradient_boosting": gb_model,

    "rf_weight": rf_weight,

    "gb_weight": gb_weight,

    "features": features
}


joblib.dump(

    model_package,

    "models/weather_blend_model.pkl"
)


print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print(
    "models/weather_blend_model.pkl"
)

print(
    "weather_predictions.csv"
)