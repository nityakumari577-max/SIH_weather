import pandas as pd

from pymongo import MongoClient

from datetime import datetime


# =====================================================
# 1. CONNECT TO MONGODB
# =====================================================

client = MongoClient(
    "mongodb://localhost:27017/"
)

db = client["weather_blending"]


# Collections

historical_collection = db[
    "historical_weather"
]

forecast_collection = db[
    "forecasts"
]

model_collection = db[
    "model_results"
]


# =====================================================
# 2. LOAD ORIGINAL DATA
# =====================================================

df = pd.read_csv(
    "data/DailyDelhiClimateTrain.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)


# =====================================================
# 3. CONVERT DATA INTO MONGODB FORMAT
# =====================================================

records = df.to_dict(
    orient="records"
)


# Convert Pandas Timestamp
# into Python datetime

for record in records:

    record["date"] = record[
        "date"
    ].to_pydatetime()


# =====================================================
# 4. STORE HISTORICAL WEATHER DATA
# =====================================================

# Delete old records
historical_collection.delete_many({})


historical_collection.insert_many(
    records
)


print(
    "Historical weather data stored:",
    len(records)
)


# =====================================================
# 5. LOAD ML RESULTS
# =====================================================

results = pd.read_csv(
    "weather_predictions.csv"
)


# =====================================================
# 6. STORE FORECASTS
# =====================================================

forecast_records = results.to_dict(
    orient="records"
)


forecast_collection.delete_many({})


forecast_collection.insert_many(
    forecast_records
)


print(
    "Forecast results stored:",
    len(forecast_records)
)


# =====================================================
# 7. STORE MODEL INFORMATION
# =====================================================

model_information = {

    "model_name":
        "Hybrid Weather ML Blending Model",

    "models": [

        "Random Forest",

        "Gradient Boosting"

    ],

    "target":
        "Next Day Mean Temperature",

    "created_at":
        datetime.now(),

    "status":
        "trained"

}


model_collection.insert_one(
    model_information
)


print(
    "Model information stored!"
)


print("\nMongoDB pipeline completed!")