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

# Set page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS to match the design
st.markdown(
    """
    <style>
        .stApp {
            background-color: #1A1A1A;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #1A1A1A;
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
            background-color: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            padding: 8px;
            color: #FFFFFF;
            box-shadow: none;
        }
        .stButton>button {
            background-color: #FF4040;
            color: #FFFFFF;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .stButton:nth-child(2)>button {
            background-color: #1A1A1A;
            border: 1px solid #FFFFFF;
        }
        .stSuccess {
            background-color: #006400;
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
            display: none;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .css-1aumxbw { /* Adjust layout for centered content */
            max-width: 100%;
            padding: 0 20px;
        }
        .stSelectbox div[role="combobox"] {
            background-color: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            height: 40px;
            padding: 0 8px;
            box-shadow: none;
        }
        .stSelectbox div[role="listbox"] {
            background-color: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            box-shadow: none;
        }
        .stSelectbox div[role="option"] {
            color: #FFFFFF;
            background-color: #2A2A2A;
            padding: 8px;
        }
        .stSelectbox div[role="option"]:hover {
            background-color: #3A3A3A;
        }
        .stSelectbox div[role="option"][aria-selected="true"] {
            background-color: #3A3A3A;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Multiple Disease Prediction System")
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
    gender = st.selectbox("Gender", options=["Female", "Male"])
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)

with col2:
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)

col3, col4 = st.columns(2)

with col3:
    cholesterol = st.selectbox("Cholesterol level", options=["Normal", "Above normal", "High"])
    gluc = st.selectbox("Glucose level", options=["Normal", "Above normal", "High"])
    smoke = st.selectbox("Do you smoke?", options=["No", "Yes"])

with col4:
    active = st.selectbox("Are you physically active?", options=["No", "Yes"])

# Predict button
if st.button("🔍 Heart Disease Test Result"):
    input_data = pd.DataFrame([{
        'age': age,
        'gender': 1 if gender == "Female" else 2,  # Map Female to 1, Male to 2
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': {"Normal": 1, "Above normal": 2, "High": 3}[cholesterol],
        'gluc': {"Normal": 1, "Above normal": 2, "High": 3}[gluc],
        'smoke': 1 if smoke == "Yes" else 0,
        'active': 1 if active == "Yes" else 0
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

# Camera button (simulated)
st.markdown(
    """
    <button class="camera-btn" onclick="alert('Camera feature not implemented')">📷</button>
    <style>
        .camera-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #000000;
            color: #FFFFFF;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 20px;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)