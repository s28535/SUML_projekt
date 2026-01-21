import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import joblib
from xgboost import XGBRegressor


def load_data():
    # pobieranie datasetów
    path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland")

    # wywalenie wszystkich plikow dotyczacych wynajmów
    for file_name in os.listdir(path):
        if "rent" in file_name.lower():
            file_path = os.path.join(path, file_name)
            os.remove(file_path)
    
    # polaczenie wszystkich pozostalych plikow w jeden
    files = os.listdir(path)
    df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    
    # zostawienie tylko tych wierszy ktore maja "city" == "warszawa"
    df = df[df['city'] == 'warszawa']
    return df


def show_basic_info(df):
    print("=" * 50)
    print("PODSTAWOWE INFORMACJE O DANYCH")
    print("=" * 50)
    print(f"\nLiczba wierszy: {len(df)}")
    print(f"\nKolumny: {list(df.columns)}")
    print("\n" + "=" * 50)
    print("PIERWSZE 5 WIERSZY:")
    print("=" * 50)
    print(df.head())
    print("\n" + "=" * 50)
    print("INFO O DANYCH:")
    print("=" * 50)
    print(df.info())
    print("\n" + "=" * 50)
    print("STATYSTYKI NUMERYCZNE:")
    print("=" * 50)
    print(df.describe())
    print("\n" + "=" * 50)
    print("BRAKI DANYCH:")
    print("=" * 50)
    missing = df.isnull().sum()
    print(missing)
    print("\n" + "=" * 50)
    print("PROCENT BRAKÓW DANYCH:")
    print("=" * 50)
    print((missing / len(df) * 100).sort_values(ascending=False))
    
    # pokazanie typów danych
    print("\n" + "=" * 50)
    print("IDENTYFIKACJA TYPÓW DANYCH:")
    print("=" * 50)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # usuniecie price
    if 'price' in num_cols:
        num_cols.remove('price')
    
    print(f"\nKolumny NUMERYCZNE ({len(num_cols)}):")
    print(num_cols)
    print(f"\nKolumny KATEGORYCZNE ({len(cat_cols)}):")
    print(cat_cols)
    print(f"\nTARGET: price")
    
    # pokazanie unikalnych wartości w kolumnach kategorycznych
    print("\n" + "=" * 50)
    print("UNIKALNE WARTOŚCI W KOLUMNACH KATEGORYCZNYCH:")
    print("=" * 50)
    for col in cat_cols:
        if col not in ['id', 'city']:  # pomijam id i city
            print(f"\n{col}:")
            print(df[col].value_counts())
            print(f"Liczba unikalnych wartości: {df[col].nunique()}")
    
    return num_cols, cat_cols


def handle_missing_values(df, num_cols, cat_cols):
    df_clean = df.copy()
    
    print("\n" + "=" * 50)
    print("OBSŁUGA BRAKÓW DANYCH:")
    print("=" * 50)
    
    # condition ma 76% braków - usuwamy kolumnę
    print("\nUsuwanie kolumny condition (76% braków)")
    if 'condition' in df_clean.columns:
        df_clean = df_clean.drop(columns=['condition'])
    
    # buildingMaterial ma 42% braków - usuwamy kolumnę
    print("Usuwanie kolumny buildingMaterial (42% braków)")
    if 'buildingMaterial' in df_clean.columns:
        df_clean = df_clean.drop(columns=['buildingMaterial'])
    
    # type ma 25% braków - tworzymy "unknown"
    print("Uzupełnianie type (25% braków) - kategoria 'unknown'")
    df_clean['type'] = df_clean['type'].fillna('unknown')
    
    # Uzupełnianie numerycznych wartości medianą
    print("\nUzupełnianie kolumn NUMERYCZNYCH medianą:")
    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            missing_count = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"  {col}: {missing_count} braków uzupełnionych medianą ({median_val:.2f})")
    
    # Uzupełnianie kategorycznych wartości modą lub "unknown"
    print("\nhasElevator: uzupełnianie braków wartością 'yes'")
    df_clean["hasElevator"] = df_clean["hasElevator"].fillna("yes")
    
    # Sprawdzenie czy są jeszcze braki
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"\nPozostałe braki danych: {remaining_missing}")
    
    return df_clean


def feature_engineering(df):
    df_fe = df.copy()
    
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING:")
    print("=" * 50)
    
    # tworzenie cech "isNear", 1 - jezeli odleglosc mniejsza niz 0.5km, 0 w przeciwnym razie
    distance_cols_mapping = {
        'schoolDistance': 'isSchoolNear',
        'clinicDistance': 'isClinicNear',
        'postOfficeDistance': 'isPostOfficeNear',
        'kindergartenDistance': 'isKindergartenNear',
        'restaurantDistance': 'isRestaurantNear',
        'collegeDistance': 'isCollegeNear',
        'pharmacyDistance': 'isPharmacyNear'
    }
    
    for distance_col, is_near_col in distance_cols_mapping.items():
        if distance_col in df_fe.columns:
            df_fe[is_near_col] = (df_fe[distance_col] < 0.5).astype(int)
            print(f"Utworzono: {is_near_col} (1 jeśli {distance_col} < 0.5, 0 w przeciwnym razie)")
    
    # usuniecie oryginalnych kolumn z odleglosciami
    distance_cols_to_remove = list(distance_cols_mapping.keys())
    df_fe = df_fe.drop(columns=distance_cols_to_remove)
    print(f"\nUsunieto kolumny odleglosci: {distance_cols_to_remove}")
    
    # tworzenie cech "building_age" - wiek budynku
    current_year = 2026
    df_fe['building_age'] = current_year - df_fe['buildYear']
    df_fe['building_age'] = df_fe['building_age'].clip(lower=0)
    print("Utworzono: building_age (wiek budynku)")
    
    # tworzenie cech "floor_ratio" - stosunek piętra do liczby pięter (pozycja w budynku)
    df_fe['floor_ratio'] = df_fe['floor'] / df_fe['floorCount']
    df_fe['floor_ratio'] = df_fe['floor_ratio'].fillna(0.5)
    print("Utworzono: floor_ratio (pozycja pietra w budynku)")
    
    # tworzenie cech "sqm_per_room" - metry kwadratowe na pokój
    df_fe['sqm_per_room'] = df_fe['squareMeters'] / df_fe['rooms']
    print("Utworzono: sqm_per_room (m2 na pokoj)")
    
    print(f"\nLiczba cech przed FE: {len(df.columns)}")
    print(f"Liczba cech po FE: {len(df_fe.columns)}")
    
    return df_fe


def prepare_data_for_modeling(df):
    df_prep = df.copy()
    
    print("\n" + "=" * 50)
    print("PRZYGOTOWANIE DANYCH DO MODELOWANIA:")
    print("=" * 50)
    
    # kolumny do usuniecia
    cols_to_drop = ['id', 'city', 'price']
    
    # zmiana kolumn yes/no na 0/1
    yes_no_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    print("\nzmiana kolumn yes/no na 0/1:")
    for col in yes_no_cols:
        df_prep[col] = df_prep[col].map({'yes': 1, 'no': 0})
        print(f"  {col}: yes->1, no->0")
    
    # One-Hot Encoding dla pozostałych kategorycznych
    # buildingMaterial i condition zostały usunięte (za dużo braków)
    categorical_cols = ['type', 'ownership']
    
    print(f"\nOne-Hot Encoding dla kolumn: {categorical_cols}")
    df_prep = pd.get_dummies(df_prep, columns=categorical_cols, drop_first=True)
    
    # przygotowanie X i y
    y = df_prep['price']
    X = df_prep.drop(columns=cols_to_drop)
    
    return X, y, df_prep


def train_and_evaluate_models(X, y):
    # Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nPodział danych:")
    print(f"  Train: {X_train.shape[0]} próbek")
    print(f"  Test: {X_test.shape[0]} próbek")
    
    # Skalowanie danych (dla modeli, które tego wymagają)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression (scaled)': LinearRegression(),
        'Lasso (scaled)': Lasso(alpha=1.0, random_state=42),
        'Ridge (scaled)': Ridge(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    print("\n" + "=" * 50)
    print("TRENOWANIE MODELI:")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\nTrenowanie: {name}")
        
        if 'scaled' in name.lower():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Metryki
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
        
        results[name] = {
            'models': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  R2: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    # Wybór najlepszego modelu (najniższy RMSE)
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = results[best_model_name]['models']
    
    print("\n" + "=" * 50)
    print("PODSUMOWANIE WYNIKÓW:")
    print("=" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  RMSE: {metrics['rmse']:,.2f}")
        print(f"  MAE: {metrics['mae']:,.2f}")
        print(f"  R2: {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"NAJLEPSZY MODEL: {best_model_name}")
    print(f"RMSE: {results[best_model_name]['rmse']:,.2f}")
    print(f"R2: {results[best_model_name]['r2']:.4f}")
    print(f"MAPE: {results[best_model_name]['mape']:.2f}%")
    print(f"{'='*50}")
    
    # Analiza ważności cech dla Random Forest
    feature_importance = None
    if best_model_name == 'Random Forest':
        print("\n" + "=" * 50)
        print("ANALIZA WAZNOSCI CECH (Random Forest):")
        print("=" * 50)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nWażność cech:")
        for idx, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return best_model


def main():
    df = load_data()
    
    num_cols, cat_cols = show_basic_info(df)

    df_clean = handle_missing_values(df, num_cols, cat_cols)
    df_fe = feature_engineering(df_clean)
    X, y, df_prep = prepare_data_for_modeling(df_fe)
    
    print("\n" + "=" * 50)
    print("PODSUMOWANIE PRZYGOTOWANIA DANYCH:")
    print("=" * 50)
    print(f"Liczba próbek: {len(X)}")
    print(f"Liczba cech: {X.shape[1]}")
    
    # trenowanie modeli
    best_model = train_and_evaluate_models(X, y)
    
    # zapisanie najlepszego modelu
    print("\n" + "=" * 50)
    print("ZAPISYWANIE MODELU:")
    print("=" * 50)
    model_filename = 'apartment_price_model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model zapisany do: {model_filename}")
    
    print("\n" + "=" * 50)
    print("ZAKONCZONO TRENOWANIE MODELU")
    print("=" * 50)


if __name__ == "__main__":
    main()