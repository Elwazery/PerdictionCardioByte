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
    df['age'] = df['age_years'] * 365
    df['years'] = df['age_years']
    df = df.drop(columns=['age_years'], errors='ignore')

    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

    df['age_bin'] = pd.cut(df['years'], bins=[0, 30, 40, 50, 60, 100],
                           labels=['<30', '30-40', '40-50', '50-60', '60+'], include_lowest=True)
    df['BMI_Class'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    df['MAP_Class'] = pd.cut(df['MAP'], bins=[0, 80, 100, 120, 1000],
                             labels=['Low', 'Normal', 'High', 'Very High'], include_lowest=True)

    for col in clipping_bounds:
        df[col] = df[col].clip(lower=clipping_bounds[col]['lower'], upper=clipping_bounds[col]['upper'])

    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    df['BMI_Class'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    df['MAP_Class'] = pd.cut(df['MAP'], bins=[0, 80, 100, 120, 1000],
                             labels=['Low', 'Normal', 'High', 'Very High'], include_lowest=True)

    categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'active',
                           'age_bin', 'BMI_Class', 'MAP_Class']
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col].astype(str))

    df['Cluster'] = np.nan
    is_male = df['gender'].values[0] == label_encoders['gender'].transform(['2'])[0]
    cluster_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'active',
                    'age_bin', 'BMI_Class', 'MAP_Class']
    if is_male:
        df['Cluster'] = km_male.predict(df[cluster_cols])
    else:
        df['Cluster'] = km_female.predict(df[cluster_cols])

    features = ['years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'active', 'bmi', 'MAP',
                'age_bin', 'BMI_Class', 'MAP_Class', 'Cluster']
    return df[features]

# Streamlit App
st.set_page_config(page_title="Cardio Risk Predictor", layout="centered")

st.title("🫀 Cardiovascular Risk Prediction")
st.markdown("Fill in the fields below to predict the risk of cardiovascular disease.")

model, label_encoders, km_male, km_female, clipping_bounds = load_models()

with st.form("manual_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input("Age (in years)", min_value=1, max_value=120, step=1, value=None)
    with col2:
        gender = st.selectbox("Gender", ["", "Female", "Male"])

    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("Height (in cm)", min_value=100, max_value=250, step=1, value=None)
    with col4:
        weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, step=0.1, value=None)

    col5, col6 = st.columns(2)
    with col5:
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=50, max_value=250, step=1, value=None)
    with col6:
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=30, max_value=200, step=1, value=None)

    col7, col8 = st.columns(2)
    with col7:
        cholesterol = st.selectbox("Cholesterol Level", ["", "Normal", "Above Normal", "Well Above Normal"])
    with col8:
        gluc = st.selectbox("Glucose Level", ["", "Normal", "Above Normal", "Well Above Normal"])

    col9, col10 = st.columns(2)
    with col9:
        smoke = st.selectbox("Smoker", ["", "No", "Yes"])
    with col10:
        active = st.selectbox("Physically Active", ["", "Yes", "No"])

    submitted = st.form_submit_button("Predict")

    if submitted:
        if None in [age_years, height, weight, ap_hi, ap_lo] or \
           "" in [gender, cholesterol, gluc, smoke, active]:
            st.warning("⚠️ Please complete all fields before predicting.")
        else:
            cholesterol_map = {"Normal": "1", "Above Normal": "2", "Well Above Normal": "3"}
            gluc_map = cholesterol_map
            smoke_map = {"No": "0", "Yes": "1"}
            active_map = {"Yes": "1", "No": "0"}

            input_data = {
                'age_years': age_years,
                'gender': '2' if gender == "Male" else '1',
                'height': height,
                'weight': weight,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'cholesterol': cholesterol_map[cholesterol],
                'gluc': gluc_map[gluc],
                'smoke': smoke_map[smoke],
                'active': active_map[active]
            }

            try:
                processed_input = preprocess_manual_input(input_data, label_encoders, km_male, km_female, clipping_bounds)
                prediction = model.predict(processed_input)[0]
                probability = model.predict_proba(processed_input)[0][1]

                st.subheader("📊 Prediction Result")
                if prediction == 1:
                    st.error(f"⚠️ High risk of cardiovascular disease.\nProbability: {probability * 100:.2f}%")
                else:
                    st.success(f"✅ Low risk of cardiovascular disease.\nProbability: {probability * 100:.2f}%")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Powered by XGBoost, KModes & Streamlit")
