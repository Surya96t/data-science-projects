CFGLog = {
    "data": {
        "path": "./data/SeoulBikeData_cleaned_cols.csv",
        "X": [
            'hour', 'temp',
            'humidity', 'wind_speed', 
            'visibility', 'solar_rad',
            'rainfall', 'snowfall', 'seasons',
            'holiday', 'day', 'month'
        ],
        "y": "rented_bike_count",
        "test_size": 0.2,
        "random_state": 42
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": 7,
        "subsample": 0.8,
        "learning_rate": 0.1
    },
    "output": {
        "output_path": "./data/exported_models/",
        "xgb_model": "XGBoost_2025-01-24_14-51-46.pkl",
    }
}

