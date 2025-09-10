import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("🌾 Sustainable Agriculture Crop Yield Prediction")

rice_area = st.number_input("Enter Rice Area (1000 ha)", min_value=0.0, step=1.0)
rice_production = st.number_input("Enter Rice Production (1000 tons)", min_value=0.0, step=1.0)
year = st.number_input("Enter Year", min_value=2000, max_value=2100, step=1)

if st.button("Predict Yield"):
    features = pd.DataFrame(
        [[rice_area, rice_production, year]],
        columns=["RICE AREA (1000 ha)", "RICE PRODUCTION (1000 tons)", "Year"]
    )
    prediction = model.predict(features)
    st.success(f"🌱 Predicted Rice Yield: {prediction[0]:.2f} Kg/ha")
