import pandas as pd
import streamlit as st
import numpy as np
import joblib

st.write("""
# Welcome To The Real Estate Price Estimate In City Of Bengaluru, India

This app predicts prices of houses in estate within **Bengaluru, India**
""")
st.write('***')

# Load model and scaler
model = joblib.load('model_defense.joblib')
scaler = joblib.load('scaler.joblib')

# Load dataset
x = pd.read_csv('x_defense.csv')
x.drop('Unnamed: 0' ,axis = 1 ,inplace = True)
locations = x.columns[7:]
area_types = x.columns[3:7]

def predict_price(x, location, area, total_sqft, bath, bhk, scaler):
    total_sqft, bath, bhk = float(total_sqft), float(bath), float(bhk)

    loc_index = np.where(x.columns == location)[0]
    area_index = np.where(x.columns == area)[0]

    loc_index = loc_index[0] if loc_index.size > 0 else -1
    area_index = area_index[0] if area_index.size > 0 else -1

    fv = np.zeros(len(x.columns), dtype=np.float32)
    scaled_values = scaler.transform(np.array([[total_sqft, bath, bhk]]))[0]

    fv[0], fv[1], fv[2] = scaled_values[0], scaled_values[1], scaled_values[2]

    if loc_index >= 0:
        fv[loc_index] = 1
    if area_index >= 0:
        fv[area_index] = 1

    fv = fv.reshape(1, -1)
    return float(model.predict(fv)[0])
st.sidebar.write('Select the location and area type')

location = st.sidebar.selectbox("Select your desired location", locations)
area = st.sidebar.selectbox("Select the type of area you desire", area_types)

st.subheader('Select your choice of bedrooms, bathrooms, and square footage')

total_sqft = st.number_input("Enter the number of Square Foot(Metres)", 
                             min_value= 300 ,
                             max_value=5000,
                             value=1000)
bath = st.number_input("How many bathrooms apartment?", 
                       min_value= 1 ,
                       max_value= 15)
bhk = st.slider("How many Bedroom, Living Room Apartment?", 
                min_value= 1,
                max_value = 14)

if st.button("Estimate"):
    price = predict_price(x, location, area, total_sqft, bath, bhk, scaler) * 1140
    st.success(f"🏠 The estimated price of the house is **${price:,.2f}**")



