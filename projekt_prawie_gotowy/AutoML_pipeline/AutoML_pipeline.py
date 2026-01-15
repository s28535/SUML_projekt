from pycaret.regression import *
import kagglehub
import os
import pandas as pd
import numpy as np
import joblib

def load_data():
    path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

    for file_name in os.listdir(path):
        if "rent" in file_name.lower():
            os.remove(os.path.join(path, file_name))

    files = os.listdir(path)
    df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    df = df[df['city'] == 'warszawa']
    return df


def feature_engineering(df):
    df_fe = df.copy()

    distance_cols_mapping = {
        'schoolDistance': 'isSchoolNear',
        'clinicDistance': 'isClinicNear',
        'postOfficeDistance': 'isPostOfficeNear',
        'kindergartenDistance': 'isKindergartenNear',
        'restaurantDistance': 'isRestaurantNear',
        'collegeDistance': 'isCollegeNear',
        'pharmacyDistance': 'isPharmacyNear'
    }

    for dist, near in distance_cols_mapping.items():
        df_fe[near] = (df_fe[dist] < 0.5).astype(int)

    df_fe = df_fe.drop(columns=list(distance_cols_mapping.keys()))

    df_fe['building_age'] = 2026 - df_fe['buildYear']
    df_fe['building_age'] = df_fe['building_age'].clip(lower=0)

    df_fe['floor_ratio'] = df_fe['floor'] / df_fe['floorCount']
    df_fe['floor_ratio'] = df_fe['floor_ratio'].fillna(0.5)

    df_fe['sqm_per_room'] = df_fe['squareMeters'] / df_fe['rooms']

    return df_fe


def train_with_pycaret(df):
    print("\n===========================================")
    print("TRENING MODELU W PYCARET")
    print("===========================================\n")

    reg_setup = setup(
        data=df,
        target='price',
        session_id=42,
        normalize=True,
        remove_outliers=False,
        polynomial_features=False,
        feature_interaction=False,
        feature_ratio=False,
        silent=True,
        verbose=False
    )

    print("Trwa porównywanie modeli...")
    best_model = compare_models()

    print("\nNajlepszy model:")
    print(best_model)

    final_model = finalize_model(best_model)

    save_model(final_model, "pycaret_apartment_model")

    print("\nModel zapisany jako pycaret_apartment_model.pkl")

    return final_model


def main():
    df = load_data()

    df_fe = feature_engineering(df)

    # === PYCARET TRAINING ===
    model = train_with_pycaret(df_fe)

    print("\nZAKOŃCZONO TRENING.")


if __name__ == "__main__":
    main()
