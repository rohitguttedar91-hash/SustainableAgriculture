import streamlit as st
import pickle
import numpy as np

with open("rf_crop_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("crop_label_encoder.pkl", "rb") as f:
    le_crop = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("🌾 Crop Prediction App (Random Forest)")

st.write("Enter the agricultural and environmental details to predict the crop.")


year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
area = st.number_input("Area (in hectares)", min_value=1, value=100)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=1000.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
price = st.number_input("Price (₹)", min_value=0, value=5000)

location = st.selectbox("Location", label_encoders['Location'].classes_)
soil_type = st.selectbox("Soil type", label_encoders['Soil type'].classes_)
irrigation = st.selectbox("Irrigation", label_encoders['Irrigation'].classes_)
season = st.selectbox("Season", label_encoders['Season'].classes_)

if st.button("Predict Crop"):
    loc_enc = label_encoders['Location'].transform([location])[0]
    soil_enc = label_encoders['Soil type'].transform([soil_type])[0]
    irr_enc = label_encoders['Irrigation'].transform([irrigation])[0]
    season_enc = label_encoders['Season'].transform([season])[0]

    input_data = np.array([[year, loc_enc, area, rainfall, temperature,
                            soil_enc, irr_enc, 0.0, humidity, price, season_enc]])
    
    prediction = model.predict(input_data)[0]
    crop_name = le_crop.inverse_transform([prediction])[0]

    st.success(f"🌱 Predicted Crop: **{crop_name}**")
