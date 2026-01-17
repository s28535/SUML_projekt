import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import Model



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

def prepare_data(df):
    df = df.drop('id', axis=1)
    df = df.drop('type', axis=1)
    df = df.drop('condition', axis=1)
    df = df.drop('buildingMaterial', axis=1)

    #Drop city bo wiemy ze wsztskie wartosci jakie zostaly to warszawa
    df = df.drop('city', axis=1)

    ownership_Typedummies = pd.get_dummies(df['ownership'], prefix='', dtype=int)
    df = pd.concat([df, ownership_Typedummies], axis=1)
    df = df.drop('ownership', axis=1)


    #Zamienienie brakujacych wartosci srednia
    df['floor'] = df['floor'].fillna(df['floor'].mean())
    df['floorCount'] = df['floorCount'].fillna(df['floorCount'].mean())
    df['buildYear'] = df['buildYear'].fillna(df['buildYear'].mean())
    df['schoolDistance'] = df['schoolDistance'].fillna(df['schoolDistance'].mean())
    df['clinicDistance'] = df['clinicDistance'].fillna(df['clinicDistance'].mean())

    df['postOfficeDistance'] = df['postOfficeDistance'].fillna(df['postOfficeDistance'].mean())
    df['kindergartenDistance'] = df['kindergartenDistance'].fillna(df['kindergartenDistance'].mean())
    df['restaurantDistance'] = df['restaurantDistance'].fillna(df['restaurantDistance'].mean())
    df['collegeDistance'] = df['collegeDistance'].fillna(df['collegeDistance'].mean())
    df['pharmacyDistance'] = df['pharmacyDistance'].fillna(df['pharmacyDistance'].mean())
    df['hasElevator'] = df['hasElevator'].fillna(df['hasElevator'].mean())

    for column in df.columns:
        missingCount = df[column].isna().sum()
        print(f"Potencjalnie usuniete wiersze '{column}': {missingCount}")

    df.to_csv('data/intermediate/clean_data.csv',index=False)

    return df

def train_test_seperate(df, scaler_X, scaler_y):

    # Podzial na zbior treningowy i testowy

    #Ustawienie zmiennej celo
    X = df.drop('price', axis=1)
    y = df['price']

    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    X_train = torch.FloatTensor(X_train_scaled)
    X_test = torch.FloatTensor(X_test_scaled)

    y_train = torch.FloatTensor(y_train_scaled)
    y_test = torch.FloatTensor(y_test_scaled)


    array = X_train.numpy()
    df = pd.DataFrame(array)
    df.to_csv('data/model_input/featuresTrain.csv', index=False)

    array = y_train.numpy()
    df = pd.DataFrame(array)
    df.to_csv('data/model_input/scoreTrain.csv', index=False)

    array = X_test.numpy()
    df = pd.DataFrame(array)
    df.to_csv('data/model_input/featuresTest.csv', index=False)

    array = y_test.numpy()
    df = pd.DataFrame(array)
    df.to_csv('data/model_input/scoreTest.csv', index=False)

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


    print("Ilosc danych dla wszystkich miast: " , len(df))

    #zostawienie tylko tych wierszy ktore maja "city" == "warszawa"
    df = df[df['city'] == 'warszawa']

    print("Ilosc danych dla wszystkich Warszawy: " , len(df))

    print("Przecietna cena mieszkania", df['price'].mean())
    print(df.info())
    print()

    check_missing_values(df)
    check_object_columns(df)
    convert_yes_no(df)
    convert_to_numeric(df)
    check_correlations(df)

    df = prepare_data(df)

    scaler_X = preprocessing.StandardScaler()
    scaler_y = preprocessing.StandardScaler()

    train_test_seperate(df,scaler_X,scaler_y)
   # print(df.info())
  #  print()

    FeaturesTestDF = pd.read_csv("data/model_input/featuresTest.csv")
    FeaturesTrainDF = pd.read_csv("data/model_input/featuresTrain.csv")
    ScoreTestDF = pd.read_csv("data/model_input/scoreTest.csv")
    ScoreTrainDF = pd.read_csv("data/model_input/scoreTrain.csv")


#    model1 = Model.ModelMyNN()
 #   model1.trainModel(FeaturesTrainDF, ScoreTrainDF, FeaturesTestDF, ScoreTestDF, 101, scaler_X, scaler_y)

 #   model2 = Model.AutoML()
 #   model2.testDifrentMLM(df)

if __name__ == "__main__":
    main()
