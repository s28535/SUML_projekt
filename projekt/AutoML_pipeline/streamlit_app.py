import streamlit as st
import pandas as pd
from pathlib import Path
from pycaret.regression import load_model
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import math
import requests

MODEL_PATH = Path("projekt/AutoML_pipeline/data/models/apartment_price_model")

st.set_page_config(page_title="Kalkulator Cen Mieszkań w Warszawie (AutoML)", layout="wide")

st.title("Kalkulator Cen Mieszkań w Warszawie (AutoML)")
st.markdown("---")

@st.cache_resource
def load_trained_model():
    try:
        return load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Nie udało się wczytać modelu: {exc}")
        return None

model = load_trained_model()
if model is None:
    st.stop()

st.header("Wprowadź dane mieszkania")

centerX = 52.2354167
centerY = 21.0074722
xDistanceSquare = 0
yDistanceSquare = 0
centre_distance = 0

m = folium.Map(location=[centerX, centerY], zoom_start=12)

map = st_folium(m, width=700, height=500)

if map.get("last_clicked"):
    x = map["last_clicked"]["lat"]
    y = map["last_clicked"]["lng"]
    st.session_state.marker_location = [x, y]



    xDistanceSquare = (x - centerX)**2
    yDistanceSquare = (y - centerY)**2
    centre_distance = geodesic((centerX, centerY), (x, y)).kilometers
    st.write(f"Odległość od centrum: ", centre_distance)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Podstawowe informacje")

    square_meters = st.number_input("Powierzchnia (m²)", min_value=20.0, max_value=200.0, value=60.0, step=1.0)
    rooms = st.number_input("Liczba pokoi", min_value=1, max_value=6, value=3, step=1)

    build_year = st.number_input("Rok budowy", min_value=1900, max_value=2026, value=2000, step=1)

    floor = st.number_input("Piętro", min_value=0, max_value=50, value=3, step=1)
    floor_count = st.number_input("Liczba pięter w budynku", min_value=1, max_value=50, value=5, step=1)

    sqm_per_room = square_meters / rooms if rooms > 0 else 0
    floor_ratio = floor / floor_count if floor_count > 0 else 0.5
    building_age = 2026 - build_year

with col2:
    st.subheader("Typ i własność")

    type_building_options = {
        "Blok mieszkalny": "blockOfFlats",
        "Budynek apartamentowy": "apartmentBuilding",
        "Kamienica": "tenement",
        "Nieznany": "unknown",
    }

    ownership_options = {
        "Własność odrębna": "condominium",
        "Spółdzielcze": "cooperative",
    }

    type_building_label = st.selectbox(
        "Typ budynku",
        list(type_building_options.keys()),
        index=0,
    )

    ownership_label = st.selectbox(
        "Forma własności",
        list(ownership_options.keys()),
        index=0,
    )

    type_building = type_building_options[type_building_label]
    ownership = ownership_options[ownership_label]

st.markdown("---")

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

poi_count = sum([
    is_school_near,
    is_clinic_near,
    is_post_office_near,
    is_kindergarten_near,
    is_restaurant_near,
    is_college_near,
    is_pharmacy_near,
])

CENTER_X = 52.2354167
CENTER_Y = 21.0074722
latitude = CENTER_X
longitude = CENTER_Y

st.markdown("---")

def sendUserInputs(): return { "type": type_building, "squareMeters": square_meters, "rooms": rooms,
                               "floor": floor, "floorCount": floor_count, "buildYear": build_year,
                               "latitude": latitude, "longitude": longitude, "centreDistance": centre_distance,
                               "poiCount": poi_count, "ownership": ownership, "hasParkingSpace": "yes" if has_parking_space else "no",
                               "hasBalcony": "yes" if has_balcony else "no", "hasElevator": "yes" if has_elevator else "no",
                               "hasSecurity": "yes" if has_security else "no", "hasStorageRoom": "yes" if has_storage_room else "no",
                               "isSchoolNear": 1 if is_school_near else 0, "isClinicNear": 1 if is_clinic_near else 0,
                               "isPostOfficeNear": 1 if is_post_office_near else 0,
                               "isKindergartenNear": 1 if is_kindergarten_near else 0,
                               "isRestaurantNear": 1 if is_restaurant_near else 0, "isCollegeNear": 1 if is_college_near else 0,
                               "isPharmacyNear": 1 if is_pharmacy_near else 0, "building_age": building_age, "floor_ratio": floor_ratio,
                               "sqm_per_room": sqm_per_room, }


if st.button("Oblicz cenę", type="primary", use_container_width=True):
    try:
        predicted_price = 0
        user_data = sendUserInputs()
        print("Input nazwa/typ:")
        for k, v in user_data.items():
            print(f"{k}: {type(v).__name__}")

        response = requests.post("http://127.0.0.1:8000/predict", json=user_data)

        if response.status_code == 200:
            result = response.json()["prediction"]
            predicted_price = result
        else:
            st.error("Błąd w komunikacji z backendem.")


        st.markdown("---")
        st.header("Przewidywana cena mieszkania")

        formatted_price = f"{predicted_price:,.0f}".replace(",", " ")
        st.markdown(f"### {formatted_price} PLN")

        st.markdown("---")
        st.subheader("Wprowadzone dane:")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.write(f"**Dystans od centrum:** {centre_distance} km")
            st.write(f"**Powierzchnia:** {square_meters} m²")
            st.write(f"**Liczba pokoi:** {rooms}")
            st.write(f"**Rok budowy:** {build_year}")
            st.write(f"**Wiek budynku:** {building_age} lat")
            st.write(f"**Piętro:** {floor}/{floor_count}")

        with info_col2:
            st.write(f"**Typ budynku:** {type_building_label}")
            st.write(f"**Własność:** {ownership_label}")
            st.write(f"**Punkty POI w pobliżu:** {poi_count}/7")

    except Exception as exc:
        st.error(f"Błąd podczas obliczania ceny: {exc}")
        st.exception(exc)
