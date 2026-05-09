import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD TRAINED MODEL
# =========================

model = pickle.load(open('rf_model.pkl', 'rb'))

# load label encoders
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

# =========================
# STREAMLIT PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Housing Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Housing Price Prediction App")

st.write("Enter housing details to predict house price.")

# =========================
# USER INPUTS
# =========================

longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41.0)
total_rooms = st.number_input("Total Rooms", value=880.0)
total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
population = st.number_input("Population", value=322.0)
households = st.number_input("Households", value=126.0)
median_income = st.number_input("Median Income", value=8.3252)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
)

# =========================
# ENCODE CATEGORICAL VALUE
# =========================

ocean_encoded = label_encoders['ocean_proximity'].transform([ocean_proximity])[0]

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [ocean_encoded]
})

# =========================
# PREDICTION
# =========================

if st.button("Predict House Price"):

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

    st.balloons()