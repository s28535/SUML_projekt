import streamlit as st
from streamlit_folium import st_folium
import folium
import math
st.title("Mapa")

m = folium.Map(location=[0, 0], zoom_start=2)

centerX = 52.2354167
centerY = 21.0074722
xDistanceSquare = 0
yDistanceSquare = 0
distanceFromCenter = 0

if "marker_location" not in st.session_state:
    st.session_state.marker_location = [centerX, centerY]

m = folium.Map(location=st.session_state.marker_location, zoom_start=12)

map = st_folium(m, width=700, height=500)

if map and map.get("last_clicked"):
    x = map["last_clicked"]["lat"]
    y = map["last_clicked"]["lng"]
    st.write(f"Wybrana lokacja")
    st.write(f"Długość:" ,x)
    st.write(f"Szerokość:" ,y)


    xDistanceSquare = (x - centerX)**2
    yDistanceSquare = (y - centerY)**2
    distanceFromCenter = math.sqrt(xDistanceSquare + yDistanceSquare)
    st.write(f"Odległość od centrum: ",distanceFromCenter)




squareMeters = st.number_input("Podaj ilość metrów kwadratowych")

roomAmount = st.selectbox("Wybierz liczbę pokoi: ", ['1','2','3','4','5','6','7'])

parkingSpaces = st.checkbox("Z miejscem parkingowym")

if st.button('Oblicz cenę apartamentu'):
    calculatedPrice =((float(squareMeters) * 0.65) + (float(squareMeters) * float(roomAmount) * 0.20) + (float(squareMeters) * float(parkingSpaces) * 0.15) - (float(squareMeters) * float(distanceFromCenter))) * 17932

    st.success("Cena apartamentu: " + f"{int(calculatedPrice):,}".replace(",", " ") + " zł")


