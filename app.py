import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing objects
model = joblib.load("xgb_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
km_male = joblib.load("km_male.pkl")
km_female = joblib.load("km_female.pkl")
clipping_bounds = joblib.load("clipping_bounds.pkl")

# Import preprocessing function (place this in the same file or separate .py file)
from preprocess import preprocess_input_data  # make sure you have this defined

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered", initial_sidebar_state="collapsed")

# White background styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🫀 Heart Disease Risk Prediction")
st.markdown("Enter your medical details to predict your risk of cardiovascular disease.")

# Input form
age = st.number_input("Age (in days)", min_value=10000, max_value=25000, value=18000)
gender = st.selectbox("Gender", options={"1": "Female", "2": "Male"})
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Systolic blood pressure (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Diastolic blood pressure (ap_lo)", min_value=40, max_value=150, value=80)
cholesterol = st.selectbox("Cholesterol level", options={"1": "Normal", "2": "Above normal", "3": "High"})
gluc = st.selectbox("Glucose level", options={"1": "Normal", "2": "Above normal", "3": "High"})
smoke = st.selectbox("Do you smoke?", options={0: "No", 1: "Yes"})
alco = st.selectbox("Do you consume alcohol?", options={0: "No", 1: "Yes"})  # This will be ignored
active = st.selectbox("Are you physically active?", options={0: "No", 1: "Yes"})

if st.button("🔍 Predict"):
    input_data = pd.DataFrame([{
        'age': age,
        'gender': int(gender),
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': int(cholesterol),
        'gluc': int(gluc),
        'smoke': int(smoke),
        'alco': int(alco),
        'active': int(active)
    }])

    # Preprocess the input
    processed = preprocess_input_data(input_data, label_encoders, km_male, km_female, clipping_bounds)

    # Predict
    probability = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    st.markdown("---")
    st.subheader("🩺 Prediction Result:")
    st.write(f"🔬 Estimated risk of heart disease: **{probability * 100:.2f}%**")
    if prediction == 1:
        st.error("⚠️ Potential risk of cardiovascular disease. Please consult a medical professional.")
    else:
        st.success("✅ Low predicted risk. Stay healthy!")

