import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def check_correlations(df):
    """
    Oblicza współczynnik korelacji każdej kolumny numerycznej z kolumną 'price'
    """
    # wybieramy tylko kolumny numeryczne
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # obliczamy korelacje z kolumną 'price'
    correlations = numeric_df.corr()['price'].sort_values(ascending=False)

    print("\n=== Korelacje z kolumną 'price' ===")
    print(correlations)
    return correlations


def check_missing_values(df):
    """
    Sprawdza, ile brakujących wartości (NaN) ma każda kolumna
    """
    missing = df.isnull().sum()
    missing = missing.sort_values(ascending=False)

    print("\n=== Brakujące dane ===")
    print(missing)
    print(f"\nŁącznie kolumn z brakami: {len(missing[missing > 0])}")
    return missing


def check_object_columns(df):
    """
    Wyświetla unikalne wartości dla każdej kolumny typu object (poza 'id')
    """
    print("\n=== Wartości kolumn typu 'object' ===")
    object_cols = df.select_dtypes(include='object').columns

    for col in object_cols:
        if col != 'id':
            unique_vals = df[col].unique()
            print(f"\nKolumna: {col}")
            print(f"Liczba unikalnych wartości: {len(unique_vals)}")
            print(f"Przykładowe wartości: {unique_vals[:10]}")  # pokaż tylko 10 pierwszych


def convert_yes_no(df):
    columns_to_change = ['hasParkingSpace', 'hasBalcony', 'hasSecurity', 'hasStorageRoom', 'hasElevator']
    maping = {'yes': 1, 'no': 0}

    df[columns_to_change] = df[columns_to_change].applymap(lambda x: maping.get(x, x))


def convert_to_numeric(df):
    df['type_numerical'] = pd.factorize(df['type'])[0]
    df['ownership_numerical'] = pd.factorize(df['ownership'])[0]
    df['buildingMaterial_numerical'] = pd.factorize(df['buildingMaterial'])[0]
    df['condition_numerical'] = pd.factorize(df['condition'])[0]

    df = df.drop('type', axis=1)
    df = df.drop('ownership', axis=1)
    df = df.drop('buildingMaterial', axis=1)
    df = df.drop('condition', axis=1)


def main ():
    #pobieranie datasetów
    path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

    #wywalenie wszystkich plikow dotyczacych wynajmów
    for file_name in os.listdir(path):
        if "rent" in file_name.lower():
            file_path = os.path.join(path, file_name)
            os.remove(file_path)

    #polaczenie wszystkich pozostalych plikow w jeden
    files = os.listdir(path)
    df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    #zostawienie tylko tych wierszy ktore maja "city" == "warszawa"
    df = df[df['city'] == 'warszawa']

    print(df.info())
    print()

    check_missing_values(df)
    check_object_columns(df)
    convert_yes_no(df)
    convert_to_numeric(df)
    check_correlations(df)


if __name__ == "__main__":
    main()