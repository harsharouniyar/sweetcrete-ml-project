import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="SweetCrete ML Predictor", layout="centered")

st.title("SweetCrete Concrete Strength Predictor")
st.write("This app predicts concrete max load using the trained machine learning model.")

# Load model
@st.cache_resource
def load_model():
    with open("models/best_model_v1.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.header("Enter Concrete Mix Information")

age_days = st.number_input("Age in Days", min_value=1, max_value=365, value=28)
water_cement_ratio = st.number_input("Water-Cement Ratio", min_value=0.0, value=0.45)
pcc_cement_ratio = st.number_input("PCC-Cement Ratio", min_value=0.0, value=0.10)
density = st.number_input("Density (lb/in³)", min_value=0.0, value=0.085)
weight = st.number_input("Weight (lb)", min_value=0.0, value=1.50)

# Feature engineering
log_age = np.log(age_days)

input_data = pd.DataFrame({
    "Log_Age": [log_age],
    "WaterCement_Ratio": [water_cement_ratio],
    "PCC_Cement_Ratio": [pcc_cement_ratio],
    "Density_lb_per_in3": [density],
    "Weight_lb": [weight]
})

st.subheader("Input Data")
st.dataframe(input_data)

if st.button("Predict Max Load"):
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Max Load: {prediction:,.2f} lbf")

    st.write("This prediction is based on the trained SweetCrete ML model.")