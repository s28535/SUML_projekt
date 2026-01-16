import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import math

MODEL_NAME = 'apartment_price_model.pkl'

# Konfiguracja strony
st.set_page_config(page_title="Kalkulator Cen Mieszkań w Warszawie", layout="wide")

# Tytuł aplikacji
st.title("Kalkulator Cen Mieszkań w Warszawie")
st.markdown("---")

# Załadowanie modelu
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_NAME)
        return model
    except FileNotFoundError:
        st.error("Model nie został znaleziony. Upewnij się, że plik {MODEL_NAME} istnieje.")
        return None

model = load_model()

if model is None:
    st.stop()

# Tworzenie formularza
st.header("Wprowadź dane mieszkania")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Podstawowe informacje")
    
    square_meters = st.number_input("Powierzchnia (m²)", min_value=20.0, max_value=200.0, value=60.0, step=1.0)
    rooms = st.number_input("Liczba pokoi", min_value=1, max_value=6, value=3, step=1)
    
    build_year = st.number_input("Rok budowy", min_value=1900, max_value=2026, value=2000, step=1)
    
    floor = st.number_input("Piętro", min_value=0, max_value=50, value=3, step=1)
    floor_count = st.number_input("Liczba pięter w budynku", min_value=1, max_value=50, value=5, step=1)
    
    # Obliczenia cech pochodnych
    sqm_per_room = square_meters / rooms if rooms > 0 else 0
    floor_ratio = floor / floor_count if floor_count > 0 else 0.5
    building_age = 2026 - build_year

with col2:
    st.subheader("Typ i własność")
    
    type_building = st.selectbox(
        "Typ budynku",
        ["blockOfFlats", "apartmentBuilding", "tenement", "unknown"],
        index=0
    )
    
    ownership = st.selectbox(
        "Forma własności",
        ["condominium", "cooperative"],
        index=0
    )

st.markdown("---")

# Checkboxy dla udogodnień
st.subheader("Udogodnienia")

col3, col4 = st.columns(2)

with col3:
    has_elevator = st.checkbox("Winda", value=False)
    has_balcony = st.checkbox("Balkon", value=False)
    has_parking_space = st.checkbox("Miejsce parkingowe", value=False)

with col4:
    has_security = st.checkbox("Ochrona", value=False)
    has_storage_room = st.checkbox("Komórka lokatorska", value=False)

st.markdown("---")

# Checkboxy dla bliskości POI
st.subheader("Bliskość punktów usługowych (zaznacz jeśli bliżej niż 500m)")

col5, col6, col7 = st.columns(3)

with col5:
    is_school_near = st.checkbox("Szkoła", value=False)
    is_clinic_near = st.checkbox("Klinika", value=False)
    is_post_office_near = st.checkbox("Poczta", value=False)

with col6:
    is_kindergarten_near = st.checkbox("Przedszkole", value=False)
    is_restaurant_near = st.checkbox("Restauracja", value=False)
    is_college_near = st.checkbox("Uczelnia", value=False)

with col7:
    is_pharmacy_near = st.checkbox("Apteka", value=False)

# Obliczenie poiCount (liczba zaznaczonych checkboxów)
poi_count = sum([
    is_school_near, is_clinic_near, is_post_office_near,
    is_kindergarten_near, is_restaurant_near, is_college_near,
    is_pharmacy_near
])

CENTER_X = 52.2354167
CENTER_Y = 21.0074722
latitude = CENTER_X
longitude = CENTER_Y
centre_distance = 0

st.markdown("---")

# Przycisk do obliczenia ceny
if st.button("Oblicz cenę", type="primary", use_container_width=True):
    try:
        # One-hot encoding dla kategorycznych
        type_blockOfFlats = 1 if type_building == 'blockOfFlats' else 0
        type_tenement = 1 if type_building == 'tenement' else 0
        type_unknown = 1 if type_building == 'unknown' else 0
        
        ownership_cooperative = 1 if ownership == 'cooperative' else 0
        
        # Tworzenie DataFrame w dokładnej kolejności jaką oczekuje model
        df_input = pd.DataFrame({
            'squareMeters': [square_meters],
            'rooms': [rooms],
            'floor': [floor],
            'floorCount': [floor_count],
            'buildYear': [build_year],
            'latitude': [latitude],
            'longitude': [longitude],
            'centreDistance': [centre_distance],
            'poiCount': [poi_count],
            'hasParkingSpace': [1 if has_parking_space else 0],
            'hasBalcony': [1 if has_balcony else 0],
            'hasElevator': [1 if has_elevator else 0],
            'hasSecurity': [1 if has_security else 0],
            'hasStorageRoom': [1 if has_storage_room else 0],
            'isSchoolNear': [1 if is_school_near else 0],
            'isClinicNear': [1 if is_clinic_near else 0],
            'isPostOfficeNear': [1 if is_post_office_near else 0],
            'isKindergartenNear': [1 if is_kindergarten_near else 0],
            'isRestaurantNear': [1 if is_restaurant_near else 0],
            'isCollegeNear': [1 if is_college_near else 0],
            'isPharmacyNear': [1 if is_pharmacy_near else 0],
            'building_age': [building_age],
            'floor_ratio': [floor_ratio],
            'sqm_per_room': [sqm_per_room],
            'type_blockOfFlats': [type_blockOfFlats],
            'type_tenement': [type_tenement],
            'type_unknown': [type_unknown],
            'ownership_cooperative': [ownership_cooperative],
        })
        
        # Predykcja
        predicted_price = model.predict(df_input)[0]
        
        # Wyświetlenie wyniku
        st.markdown("---")
        st.header("Przewidywana cena mieszkania")
        
        # Formatowanie ceny
        formatted_price = f"{predicted_price:,.0f}".replace(",", " ")
        
        st.markdown(f"### {formatted_price} PLN")
        
        # Informacje dodatkowe
        st.markdown("---")
        st.subheader("Wprowadzone dane:")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"**Powierzchnia:** {square_meters} m²")
            st.write(f"**Liczba pokoi:** {rooms}")
            st.write(f"**Rok budowy:** {build_year}")
            st.write(f"**Wiek budynku:** {building_age} lat")
            st.write(f"**Piętro:** {floor}/{floor_count}")
        
        with info_col2:
            st.write(f"**Typ budynku:** {type_building}")
            st.write(f"**Własność:** {ownership}")
            st.write(f"**Punkty POI w pobliżu:** {poi_count}/7")
        
    except Exception as e:
        st.error(f"Błąd podczas obliczania ceny: {str(e)}")
        st.exception(e)
