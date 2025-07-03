import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved models and encoders
@st.cache_resource
def load_models():
    model = joblib.load("xgb_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    km_male = joblib.load("km_male.pkl")
    km_female = joblib.load("km_female.pkl")
    clipping_bounds = joblib.load("clipping_bounds.pkl")
    return model, label_encoders, km_male, km_female, clipping_bounds

# Preprocessing function
def preprocess_manual_input(data, label_encoders, km_male, km_female, clipping_bounds):
    df = pd.DataFrame([data])

    # Derived features
    df['years'] = round(df['age'] / 365).astype('int')
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

    # Binning
    df['age_bin'] = pd.cut(df['years'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'], include_lowest=True)
    df['BMI_Class'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    df['MAP_Class'] = pd.cut(df['MAP'], bins=[0, 80, 100, 120, 1000], labels=['Low', 'Normal', 'High', 'Very High'], include_lowest=True)

    # Clip outliers
    for col in clipping_bounds:
        df[col] = df[col].clip(lower=clipping_bounds[col]['lower'], upper=clipping_bounds[col]['upper'])

    # Recalculate after clipping
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    df['BMI_Class'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    df['MAP_Class'] = pd.cut(df['MAP'], bins=[0, 80, 100, 120, 1000], labels=['Low', 'Normal', 'High', 'Very High'], include_lowest=True)

    # Encode categorical columns
    categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'active', 'age_bin', 'BMI_Class', 'MAP_Class']
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col].astype(str))

    # Apply KModes clustering
    df['Cluster'] = np.nan
    is_male = df['gender'].values[0] == label_encoders['gender'].transform(['2'])[0]
    columns_for_clustering = ['gender', 'cholesterol', 'gluc', 'smoke', 'active', 'age_bin', 'BMI_Class', 'MAP_Class']
    if is_male:
        df['Cluster'] = km_male.predict(df[columns_for_clustering])
    else:
        df['Cluster'] = km_female.predict(df[columns_for_clustering])

    # Final feature list
    features = ['years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'active', 'bmi', 'MAP',
                'age_bin', 'BMI_Class', 'MAP_Class', 'Cluster']
    return df[features]

# --------------------- Streamlit App --------------------- #
st.set_page_config(page_title="Cardio Risk Predictor", layout="centered")

st.title("🫀 Cardiovascular Risk Prediction")
st.markdown("Fill in the fields below to predict the risk of cardiovascular disease.")

# Load models
model, label_encoders, km_male, km_female, clipping_bounds = load_models()

with st.form("manual_input_form"):
    age = st.number_input("Age (in days)", min_value=365, max_value=36500, value=18250)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, value=80)
    cholesterol = st.selectbox("Cholesterol", ["1", "2", "3"])
    gluc = st.selectbox("Glucose", ["1", "2", "3"])
    smoke = st.selectbox("Smoker", ["0", "1"])
    active = st.selectbox("Physically Active", ["0", "1"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare the input dictionary
        input_data = {
            'age': age,
            'gender': '2' if gender == "Male" else '1',
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'active': active
        }

        try:
            # Preprocess and predict
            processed_input = preprocess_manual_input(input_data, label_encoders, km_male, km_female, clipping_bounds)
            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0][1]

            st.subheader("📊 Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ High risk of cardiovascular disease.\nProbability: {probability:.2f}")
            else:
                st.success(f"✅ Low risk of cardiovascular disease.\nProbability: {probability:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Powered by XGBoost, KModes & Streamlit")
