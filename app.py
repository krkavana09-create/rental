import streamlit as st
import pandas as pd
import joblib


import joblib

MODEL_PATH = r"C:\Users\krkav\Desktop\rental listing\RentModel_new.joblib"

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="RentHop Interest Predictor", page_icon="🏠")

st.title("🏠 Rental Listing Interest Predictor")
st.caption("Educational demonstration using the RentHop rental-listing dataset.")

bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, 0.5)
bedrooms = st.number_input("Bedrooms", 0, 10, 2, 1)
price = st.number_input("Monthly Price", 0, 100000, 3000, 100)
latitude = st.number_input("Latitude", value=40.75, format="%.6f")
longitude = st.number_input("Longitude", value=-73.98, format="%.6f")
feature_count = st.number_input("Number of listed features", 0, 50, 8, 1)
photo_count = st.number_input("Number of photos", 0, 100, 6, 1)
listing_hour = st.number_input("Listing hour", 0, 23, 12, 1)
listing_dayofweek = st.number_input("Day of week (0=Mon, 6=Sun)", 0, 6, 2, 1)

description = st.text_area(
    "Rental description",
    "Beautiful renovated apartment with hardwood floors, elevator, laundry and subway access."
)
features_t0_test=st.text_area(
    "features (space-separated)",
    "Doorman Elevator Laundry Hardwood Floors Laundary"
)

if st.button("Predict Interest Level"):
    row = pd.DataFrame([{
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "price": price,
        "latitude": latitude,
        "longitude": longitude,
        "feature_count": feature_count,
        "photo_count": photo_count,
        "listing_hour": listing_hour,
        "listing_dayofweek": listing_dayofweek,
        "features_t0_test": features_t0_test,
        "description": description
    }])
    row["features_text"] = description
    prediction = model.predict(row)[0]
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]

    st.success(f"Predicted interest level: {str(prediction).upper()}")

    probability_df = pd.DataFrame({
        "Interest Level": model.classes_,
        "Probability": [f"{p:.2%}" for p in probabilities]
    })

    st.table(probability_df)

st.info("Educational demonstration only. Predictions should not be treated as guaranteed market outcomes.")
