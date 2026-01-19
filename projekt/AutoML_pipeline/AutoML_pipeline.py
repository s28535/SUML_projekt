from pycaret.regression import *
import kagglehub
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime


def load_data():
    path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

    for file_name in os.listdir(path):
        if "rent" in file_name.lower():
            os.remove(os.path.join(path, file_name))

    files = os.listdir(path)
    df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    df = df[df["city"] == "warszawa"]
    return df


def feature_engineering(df):
    df_fe = df.copy()

    distance_cols_mapping = {
        "schoolDistance": "isSchoolNear",
        "clinicDistance": "isClinicNear",
        "postOfficeDistance": "isPostOfficeNear",
        "kindergartenDistance": "isKindergartenNear",
        "restaurantDistance": "isRestaurantNear",
        "collegeDistance": "isCollegeNear",
        "pharmacyDistance": "isPharmacyNear",
    }

    for dist, near in distance_cols_mapping.items():
        df_fe[near] = (df_fe[dist] < 0.5).astype(int)

    df_fe = df_fe.drop(columns=list(distance_cols_mapping.keys()))

    df_fe["building_age"] = 2026 - df_fe["buildYear"]
    df_fe["building_age"] = df_fe["building_age"].clip(lower=0)

    df_fe["floor_ratio"] = df_fe["floor"] / df_fe["floorCount"]
    df_fe["floor_ratio"] = df_fe["floor_ratio"].fillna(0.5)

    df_fe["sqm_per_room"] = df_fe["squareMeters"] / df_fe["rooms"]

    cols_to_drop = ["id", "city", "buildingMaterial", "condition"]

    df_fe = df_fe.drop(columns=cols_to_drop)

    return df_fe


def train_with_pycaret(df):

    # Podział na zbiór treningowy i testowy
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print("Treningowy zbiór danych:")
    print(train_df.head())

    # PyCaret setup
    setup(data=train_df, target="price", session_id=42)

    # Przygotowanie katalogów wyjściowych
    model_dir = Path("data/model")
    metrics_dir = Path("data/metrics")
    charts_dir = Path("data/charts")
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Porównanie modeli
    best = compare_models()
    print(f"Najlepszy model: {best}")

    # Generowanie wykresów
    plots = ["feature", "feature_all"]

    for p in plots:
        plot_model(best, plot=p, save=str(charts_dir), display_format="streamlit")

    # Finalizacja modelu
    final_model = finalize_model(best)

    # Walidacja krzyżowa
    cv_scores = cross_val_score(
        final_model,
        train_df.drop("price", axis=1),
        train_df["price"],
        cv=5,
        scoring="r2",
    )
    print(f"Walidacja krzyżowa: {cv_scores}, średnia: {np.mean(cv_scores)}")

    # Testowanie na zbiorze testowym
    X_test = test_df.drop("price", axis=1)
    y_test = test_df["price"]
    y_pred = final_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"Test Metrics -> RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

    # Zapis wyników do JSON
    test_metrics = {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape)}
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    save_model(final_model, model_dir / "apartment_price_model")
    print("Model zapisany:", "apartment_price_model")


def main():
    df = load_data()

    df_fe = feature_engineering(df)

    model = train_with_pycaret(df_fe)

    print("\nZAKOŃCZONO TRENING.")


if __name__ == "__main__":
    main()
