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

# Import preprocessing function
from preprocess import preprocess_input_data

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #1C2526;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #1C2526;
            color: #FFFFFF;
        }
        .sidebar .stRadio label {
            color: #FFFFFF;
        }
        .sidebar .stRadio [type="radio"]:checked + span {
            background-color: #FF4040;
            color: #FFFFFF;
        }
        .stNumberInput input, .stSelectbox div {
            background-color: #2D2D2D;
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            padding: 8px;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #FF4040;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .stSuccess {
            background-color: #228B22;
            color: #FFFFFF;
            border: 1px solid #2E8B57;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .stError {
            background-color: #8B0000;
            color: #FFFFFF;
            border: 1px solid #B22222;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        .css-1cpxx8g { /* Close button styling */
            color: #FF4040;
            font-size: 20px;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation with close button
st.sidebar.title("Multiple Disease Prediction System")
st.sidebar.markdown('<a href="#" class="css-1cpxx8g">×</a>', unsafe_allow_html=True)
pages = {
    "Diabetes Prediction": "diabetes",
    "Heart Disease Prediction": "heart",
    "Pneumonia Detection": "pneumonia"
}
choice = st.sidebar.radio("", list(pages.keys()), index=1, format_func=lambda x: f"{'❤️' if x == 'Heart Disease Prediction' else ''} {x}")
if choice != "Heart Disease Prediction":
    st.sidebar.warning("This app is currently set for Heart Disease Prediction.")
    st.stop()

# Main content
st.title("❤️ Heart Disease Prediction using ML")
st.markdown("Enter your medical details to predict your risk of cardiovascular disease.")

# Navigation buttons at the top
col_nav = st.columns([1, 1, 1])
with col_nav[0]:
    if st.button("🏠 HOME"):
        st.write("Redirect to Home (not implemented)")
with col_nav[1]:
    if st.button("MULTIPLE DISEASE PREDICTION", key="multiple"):
        st.write("Redirect to Multiple Disease Prediction (not implemented)")
with col_nav[2]:
    if st.button("FEEDBACK"):
        st.write("Redirect to Feedback (not implemented)")

# Form inputs in two columns
col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (in years)", min_value=1, max_value=100, value=30)
    age = age_years * 365
    gender = st.selectbox("Gender", options={"1": "Female", "2": "Male"})
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)

with col2:
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)

col3, col4 = st.columns(2)

with col3:
    cholesterol = st.selectbox("Cholesterol level", options={"1": "Normal", "2": "Above normal", "3": "High"})
    gluc = st.selectbox("Glucose level", options={"1": "Normal", "2": "Above normal", "3": "High"})
    smoke = st.selectbox("Do you smoke?", options={0: "No", 1: "Yes"})

with col4:
    active = st.selectbox("Are you physically active?", options={0: "No", 1: "Yes"})

# Predict button
if st.button("🔍 Heart Disease Test Result"):
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
        'active': int(active)
    }])

    # Preprocess
    processed = preprocess_input_data(input_data, label_encoders, km_male, km_female, clipping_bounds)

    # Predict
    probability = model.predict_proba(processed)[0][1]
    prediction = model.predict(processed)[0]

    # Output
    st.markdown("---")
    st.subheader("Prediction Result:")
    st.write(f"Estimated risk of heart disease: **{probability * 100:.2f}%**")
    if prediction == 1:
        st.error("Potential risk of cardiovascular disease. Please consult a medical professional.")
    else:
        st.success("Low predicted risk. Stay healthy!")