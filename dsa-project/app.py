import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained models and preprocessing objects
rf_yield_model = joblib.load("rf_model.pkl")
pca_yield = joblib.load("pca_transform.pkl")
scaler_yield = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

rf_recommendation_model = joblib.load("rf_recommendation.pkl")
pca_recommendation = joblib.load("pca_recommendation.pkl")
scaler_recommendation = joblib.load("scaler_recommendation.pkl")

import base64

# Function to encode image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_bytes = base64.b64encode(image_file.read()).decode()
    return base64_bytes

# Load the image and convert it to base64
image_path = "C:/Users/rp122/OneDrive/Documents/6th Sem/data-science/python/data-science-lab/dsa-project/image.jpg"
base64_image = get_base64_of_image(image_path)

# Apply background using base64
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .output-box {{
            background-color: rgba(255, 255, 255, 1.0);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            text-align: center;
            font-size: 18px;
            color: #333;
            font-weight: bold;
            margin-top: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
# UI Title
st.title("🌾 Agriculture Prediction System")

# Sidebar for model selection
option = st.sidebar.radio("Select Model:", ["Crop Yield Prediction", "Crop Recommendation System"])

if option == "Crop Yield Prediction":
    st.header("🌱 Crop Yield Prediction System")

    # Categorical options
    crop_classes = ['Arecanut', 'Arhar/Tur', 'Castor seed', 'Coconut', 'Cotton(lint)',
           'Dry chillies', 'Gram', 'Jute', 'Linseed', 'Maize', 'Mesta',
           'Niger seed', 'Onion', 'Other Rabi pulses', 'Potato',
           'Rapeseed &Mustard', 'Rice', 'Sesamum', 'Small millets',
           'Sugarcane', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric',
           'Wheat', 'Bajra', 'Black pepper', 'Cardamom', 'Coriander',
           'Garlic', 'Ginger', 'Groundnut', 'Horse-gram', 'Jowar', 'Ragi',
           'Cashewnut', 'Banana', 'Soyabean', 'Barley', 'Khesari', 'Masoor',
           'Moong(Green Gram)', 'Other Kharif pulses', 'Safflower',
           'Sannhamp', 'Sunflower', 'Urad', 'Peas & beans (Pulses)',
           'other oilseeds', 'Other Cereals', 'Cowpea(Lobia)',
           'Oilseeds total', 'Guar seed', 'Other Summer Pulses', 'Moth']

    season_classes = ['Whole Year ', 'Kharif     ', 'Rabi       ', 'Autumn     ','Summer     ', 'Winter     ']
    state_classes = ['Assam', 'Karnataka', 'Kerala', 'Meghalaya', 'West Bengal',
           'Puducherry', 'Goa', 'Andhra Pradesh', 'Tamil Nadu', 'Odisha',
           'Bihar', 'Gujarat', 'Madhya Pradesh', 'Maharashtra', 'Mizoram',
           'Punjab', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh',
           'Tripura', 'Nagaland', 'Chhattisgarh', 'Uttarakhand', 'Jharkhand',
           'Delhi', 'Manipur', 'Jammu and Kashmir', 'Telangana',
           'Arunachal Pradesh', 'Sikkim']

    # User Inputs
    crop = st.selectbox("Select Crop:", crop_classes)
    season = st.selectbox("Select Season:", season_classes)
    state = st.selectbox("Select State:", state_classes)
    crop_year = st.number_input("Enter Crop Year:", min_value=2000, max_value=2030, value=2024)
    area = st.number_input("Enter Area (in hectares):", min_value=0.1, value=1.0)
    production = st.number_input("Enter Production (tons):", min_value=0.1, value=1.0)
    rainfall = st.number_input("Enter Annual Rainfall (mm):", min_value=0.0, value=1000.0)
    fertilizer = st.number_input("Enter Fertilizer Used (kg/ha):", min_value=0.0, value=50.0)
    pesticide = st.number_input("Enter Pesticide Used (kg/ha):", min_value=0.0, value=10.0)

    if st.button("Predict Yield"):
        try:
            # Convert categorical inputs
            crop_encoded = label_encoders['Crop'].transform([crop])[0]
            season_encoded = label_encoders['Season'].transform([season])[0]
            state_encoded = label_encoders['State'].transform([state])[0]

            # Prepare input array
            features = np.array([[crop_encoded, season_encoded, state_encoded, crop_year, area, production, rainfall, fertilizer, pesticide]])
            features[:, 3:] = scaler_yield.transform(features[:, 3:])  
            features_pca = pca_yield.transform(features)

            # Predict yield
            predicted_yield = rf_yield_model.predict(features_pca)[0]
            st.markdown(f'<div class="output-box">🌱 {predicted_yield:.2f} tons per hectare</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {str(e)}")

elif option == "Crop Recommendation System":
    st.header("🌿 Crop Recommendation System")

    # User Inputs
    nitrogen = st.number_input("Enter Nitrogen Content:", min_value=0.0, value=50.0)
    phosphorus = st.number_input("Enter Phosphorus Content:", min_value=0.0, value=50.0)
    potassium = st.number_input("Enter Potassium Content:", min_value=0.0, value=50.0)
    temperature = st.number_input("Enter Temperature (°C):", min_value=0.0, value=25.0)
    humidity = st.number_input("Enter Humidity (%):", min_value=0.0, max_value=100.0, value=60.0)
    ph_value = st.number_input("Enter Soil pH Value:", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Enter Rainfall (mm):", min_value=0.0, value=200.0)

    if st.button("Recommend Crop"):
        try:
            # Prepare input array
            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
            features_scaled = scaler_recommendation.transform(features)
            features_pca = pca_recommendation.transform(features_scaled)

            # Predict recommended crop
            recommended_crop = rf_recommendation_model.predict(features_pca)[0]
            st.markdown(f'<div class="output-box">🌾 Recommended Crop: {recommended_crop}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {str(e)}")