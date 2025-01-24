import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

st.title('Overfishing Risk Prediction Based on Fish and Environmental Data')

# Brief description
st.markdown("""
    This application allows you to input data about fish species, region, fishing methods, 
    and environmental conditions, and predicts whether there is a high risk of overfishing.
    """)

# Load pre-trained model
model = pickle.load(open("random_forest_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Form for user input
with st.form(key='overfishing_form'):
    species_name = st.selectbox('Species Name',['Salmon', 'Tuna', 'Cod', 'Herring', 'Mackerel' ,'Sardine' ,'Shark' ,'Snapper'])  # Use actual species names here
    region = st.selectbox('Region',['North Atlantic', 'Pacific Ocean', 'Mediterranean Sea', 'Indian Ocean'])  # Use actual regions
    breeding_season = st.selectbox('Breeding Season',['Summer', 'Monsoon', 'Winter'])
    fishing_method = st.selectbox('Fishing Method',['Net', 'Line', 'Trawl'])
    fish_population = st.slider('Fish Population', 0, 1000000, 500000)
    avg_size = st.slider('Average Size (cm)', 0, 200, 50)
    water_temp = st.slider('Water Temperature (°C)', 0, 40, 25)
    water_pollution = st.selectbox('Water Pollution Level',['Low', 'Medium', 'High'])

    # Submit button
    submit_button = st.form_submit_button(label='Predict Overfishing Risk')

if submit_button:
    # Display the input data
    st.write(f'### Species: {species_name}')
    st.write(f'*Region:* {region}')
    st.write(f'*Breeding Season:* {breeding_season}')
    st.write(f'*Fishing Method:* {fishing_method}')
    st.write(f'*Fish Population:* {fish_population}')
    st.write(f'*Average Size:* {avg_size} cm')
    st.write(f'*Water Temperature:* {water_temp} °C')
    st.write(f'*Water Pollution Level:* {water_pollution}')

    # Encode categorical inputs
    species_encoded = label_encoders['Species_Name'].transform([species_name])[0]
    region_encoded = label_encoders['Region'].transform([region])[0]
    breeding_season_encoded = label_encoders['Breeding_Season'].transform([breeding_season])[0]
    fishing_method_encoded = label_encoders['Fishing_Method'].transform([fishing_method])[0]
    water_pollution_encoded = label_encoders['Water_Pollution_Level'].transform([water_pollution])[0]

    # Prepare the input data for prediction
    input_data = np.array([[species_encoded, region_encoded, breeding_season_encoded, fishing_method_encoded, 
                            fish_population, avg_size, water_temp, water_pollution_encoded]])

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display prediction result
    if prediction[0] == 1:
        st.write("### Prediction: Overfishing Risk is HIGH (Yes).")
    else:
        st.write("### Prediction: Overfishing Risk is LOW (No).")