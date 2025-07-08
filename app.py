import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# Load saved models and encoders
@st.cache_resource
def load_models():
    try:
        model_cardio = joblib.load("xgb_model.pkl")
        label_encoders_cardio = joblib.load("label_encoders.pkl")
        km_male = joblib.load("km_male.pkl")
        km_female = joblib.load("km_female.pkl")
        clipping_bounds = joblib.load("clipping_bounds.pkl")
        model_diabetes = joblib.load("diabetes_final_model.pkl")
        
        # Test model compatibility
        _ = model_cardio.predict(np.zeros((1, model_cardio.n_features_in_)))
        _ = model_diabetes.predict(np.zeros((1, model_diabetes.n_features_in_)))
        
        st.sidebar.success("Models loaded successfully!")
        return model_cardio, label_encoders_cardio, km_male, km_female, clipping_bounds, model_diabetes
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}. Please ensure all model files are in the project directory.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}. Please ensure all model files are present and compatible.")
        return None, None, None, None, None, None

# Preprocessing function for cardiovascular risk
def preprocess_manual_input(data, label_encoders, km_male, km_female, clipping_bounds):
    if any(x is None for x in [label_encoders, km_male, km_female, clipping_bounds]):
        st.error("Preprocessing failed due to missing models or data. Please check the logs.")
        return pd.DataFrame()
    
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

# Preprocessing function for diabetes risk
def preprocess_diabetes_input(data, clipping_bounds):
    if clipping_bounds is None:
        st.error("Preprocessing failed due to missing clipping bounds. Please check the logs.")
        return pd.DataFrame()
    
    df = pd.DataFrame([data])
    df['age_bin'] = pd.cut(df['Age'], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                           labels=['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'], include_lowest=True)
    rating = []
    for bmi in df['BMI']:
        if bmi < 18.5:
            rating.append(1)  # Underweight
        elif bmi <= 24.9:
            rating.append(2)  # Normal
        elif bmi <= 29.9:
            rating.append(3)  # Overweight
        elif bmi <= 34.9:
            rating.append(4)  # Obesity Class 1
        elif bmi <= 39.9:
            rating.append(5)  # Obesity Class 2
        elif bmi <= 49.9:
            rating.append(6)  # Obesity Class 3
        else:
            rating.append('Error')
    df['BMI_Class'] = rating

    for col in clipping_bounds:
        if col in df.columns:
            df[col] = df[col].clip(lower=clipping_bounds[col]['lower'], upper=clipping_bounds[col]['upper'])

    features = ['Age', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'age_bin', 'BMI_Class']
    return df[features]

# Function to get top 5 influential parameters
def get_top_influences(model, processed_input, feature_names):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    top_5 = feature_importance.head(5)
    total_importance = top_5['importance'].sum()
    top_5['contribution_percent'] = (top_5['importance'] / total_importance) * 100
    return top_5

# Streamlit App
st.set_page_config(page_title="Health Risk Predictor", layout="centered")

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Cardiovascular Risk", "Diabetes Risk"],
        icons=["heart", "activity"],
        menu_icon="cast",
        default_index=0,
    )

# Load models
model_cardio, label_encoders_cardio, km_male, km_female, clipping_bounds, model_diabetes = load_models()

# Check if models loaded successfully
if any(x is None for x in [model_cardio, label_encoders_cardio, km_male, km_female, clipping_bounds, model_diabetes]):
    st.error("One or more models failed to load. Please check the error messages above and ensure all required model files are correctly uploaded.")
else:
    # Main app logic with navigation
    if selected == "Cardiovascular Risk":
        st.title("🫀 Cardiovascular Risk Prediction")
        st.markdown("Fill in the fields below to predict the risk of cardiovascular disease.")

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
                        processed_input = preprocess_manual_input(input_data, label_encoders_cardio, km_male, km_female, clipping_bounds)
                        if processed_input.empty:
                            st.error("Preprocessing failed. Please check the input data.")
                        else:
                            prediction = model_cardio.predict(processed_input)[0]
                            probability = model_cardio.predict_proba(processed_input)[0][1] * 100

                            st.subheader("📊 Prediction Result")
                            if prediction == 1:
                                st.error(f"⚠️ High risk of cardiovascular disease.\nProbability: {probability:.2f}%")
                            else:
                                st.success(f"✅ Low risk of cardiovascular disease.\nProbability: {probability:.2f}%")

                            # Get top 5 influential parameters
                            feature_names = ['years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                                             'cholesterol', 'gluc', 'smoke', 'active', 'bmi', 'MAP',
                                             'age_bin', 'BMI_Class', 'MAP_Class', 'Cluster']
                            top_influences = get_top_influences(model_cardio, processed_input, feature_names)
                            st.subheader("🔍 Top 5 Influential Parameters")
                            for _, row in top_influences.iterrows():
                                contribution = (row['contribution_percent'] / 100) * probability
                                st.write(f"{row['feature']}: {contribution:.2f}%")

                    except ValueError as ve:
                        st.error(f"Input dimensions mismatch: {ve}. Ensure input matches model features.")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

    elif selected == "Diabetes Risk":
        st.title("🍬 Diabetes Risk Prediction")
        st.markdown("Fill in the fields below to predict the risk of diabetes.")

        with st.form("diabetes_input_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age (in years)", min_value=18, max_value=100, step=1, value=None)
            with col2:
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, value=None)

            col3, col4 = st.columns(2)
            with col3:
                high_bp = st.selectbox("High Blood Pressure", ["", "No", "Yes"])
            with col4:
                high_chol = st.selectbox("High Cholesterol", ["", "No", "Yes"])

            col5, col6 = st.columns(2)
            with col5:
                smoker = st.selectbox("Smoker", ["", "No", "Yes"])
            with col6:
                phys_activity = st.selectbox("Physically Active", ["", "No", "Yes"])

            submitted = st.form_submit_button("Predict")

            if submitted:
                if None in [age, bmi] or "" in [high_bp, high_chol, smoker, phys_activity]:
                    st.warning("⚠️ Please complete all fields before predicting.")
                else:
                    high_bp_map = {"No": 0, "Yes": 1}
                    high_chol_map = {"No": 0, "Yes": 1}
                    smoker_map = {"No": 0, "Yes": 1}
                    phys_activity_map = {"No": 0, "Yes": 1}

                    input_data = {
                        'Age': age,
                        'BMI': bmi,
                        'HighBP': high_bp_map[high_bp],
                        'HighChol': high_chol_map[high_chol],
                        'Smoker': smoker_map[smoker],
                        'PhysActivity': phys_activity_map[phys_activity]
                    }

                    try:
                        processed_input = preprocess_diabetes_input(input_data, clipping_bounds)
                        if processed_input.empty:
                            st.error("Preprocessing failed. Please check the input data.")
                        else:
                            prediction = model_diabetes.predict(processed_input)[0]
                            probability = model_diabetes.predict_proba(processed_input)[0][1] * 100

                            st.subheader("📊 Prediction Result")
                            if prediction == 1:
                                st.error(f"⚠️ High risk of diabetes.\nProbability: {probability:.2f}%")
                            else:
                                st.success(f"✅ Low risk of diabetes.\nProbability: {probability:.2f}%")

                            # Get top 5 influential parameters
                            feature_names = ['Age', 'BMI', 'HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'age_bin', 'BMI_Class']
                            top_influences = get_top_influences(model_diabetes, processed_input, feature_names)
                            st.subheader("🔍 Top 5 Influential Parameters")
                            for _, row in top_influences.iterrows():
                                contribution = (row['contribution_percent'] / 100) * probability
                                st.write(f"{row['feature']}: {contribution:.2f}%")

                    except ValueError as ve:
                        st.error(f"Input dimensions mismatch: {ve}. Ensure input matches model features.")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.caption("Powered by XGBoost, KModes & Streamlit for Cardiovascular and Diabetes Prediction")
