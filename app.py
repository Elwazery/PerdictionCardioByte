import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# Load all saved models and preprocessing objects
@st.cache_resource
def load_models():
    model = joblib.load("xgb_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    km_male = joblib.load("km_male.pkl")
    km_female = joblib.load("km_female.pkl")
    clipping_bounds = joblib.load("clipping_bounds.pkl")
    return model, label_encoders, km_male, km_female, clipping_bounds

# Preprocessing function
def preprocess_input_data(input_df, label_encoders, km_male, km_female, clipping_bounds):
    df = input_df.copy()
    
    # Drop unnecessary columns
    columns_to_drop = [col for col in ['id', 'alco'] if col in df.columns]
    df = df.drop(columns_to_drop, axis=1)
    
    # Calculate derived features
    df['years'] = (df['age'] / 365).round().astype('int')
    df = df.drop('age', axis=1)
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    
    # Categorize features
    bins_age = [0, 30, 40, 50, 60, 100]
    labels_age = ['<30', '30-40', '40-50', '50-60', '60+']
    df['age_bin'] = pd.cut(df['years'], bins=bins_age, labels=labels_age, include_lowest=True)

    bins_bmi = [0, 18.5, 25, 30, 100]
    labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Class'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, include_lowest=True)

    bins_map = [0, 80, 100, 120, 1000]
    labels_map = ['Low', 'Normal', 'High', 'Very High']
    df['MAP_Class'] = pd.cut(df['MAP'], bins=bins_map, labels=labels_map, include_lowest=True)

    # Clip outliers
    for col in clipping_bounds:
        df[col] = df[col].clip(lower=clipping_bounds[col]['lower'], upper=clipping_bounds[col]['upper'])
    
    # Recalculate after clipping
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    df['BMI_Class'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, include_lowest=True)
    df['MAP_Class'] = pd.cut(df['MAP'], bins=bins_map, labels=labels_map, include_lowest=True)

    # Encode categorical values
    categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'active', 'age_bin', 'BMI_Class', 'MAP_Class']
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col].astype(str))
    
    # Predict clusters using K-Modes
    df['Cluster'] = np.nan
    male_idx = df['gender'] == label_encoders['gender'].transform(['2'])[0]
    female_idx = df['gender'] == label_encoders['gender'].transform(['1'])[0]

    columns_for_clustering = ['gender', 'cholesterol', 'gluc', 'smoke', 'active', 'age_bin', 'BMI_Class', 'MAP_Class']
    if km_male is not None and len(df[male_idx]) > 0:
        df.loc[male_idx, 'Cluster'] = km_male.predict(df[male_idx][columns_for_clustering])
    if km_female is not None and len(df[female_idx]) > 0:
        df.loc[female_idx, 'Cluster'] = km_female.predict(df[female_idx][columns_for_clustering])
    
    features = ['years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'active', 'bmi', 'MAP', 'age_bin', 'BMI_Class', 'MAP_Class', 'Cluster']
    return df[features]

# ----------------- Streamlit App ----------------- #
st.set_page_config(page_title="Cardiovascular Risk Predictor", layout="wide")

st.title("🫀 Cardiovascular Risk Prediction App")

st.markdown("Upload a CSV file with patient data to predict the risk of cardiovascular disease.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load models and encoders
    model, label_encoders, km_male, km_female, clipping_bounds = load_models()
    
    try:
        df_input = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and read successfully!")

        # Preprocess data
        df_processed = preprocess_input_data(df_input, label_encoders, km_male, km_female, clipping_bounds)

        # Make predictions
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]

        # Combine results
        results_df = df_input.copy()
        results_df['Prediction'] = predictions
        results_df['Probability_Cardio_1'] = probabilities

        # Display results
        st.subheader("📋 Prediction Results")
        st.dataframe(results_df)

        # Download button
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ An error occurred while processing the file: {e}")

else:
    st.info("Please upload a CSV file to get started.")

st.markdown("---")
st.markdown("Created by your ML pipeline 🔍 | Powered by XGBoost + KModes + Streamlit")
