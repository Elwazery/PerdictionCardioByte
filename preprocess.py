import pandas as pd
import numpy as np

def preprocess_input_data(input_df, label_encoders, km_male, km_female, clipping_bounds):
    df = input_df.copy()
    df = df.drop(['alco'], axis=1, errors='ignore')
    df['years'] = (df['age'] / 365).round().astype('int')
    df = df.drop('age', axis=1)
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3

    # Binning
    bins = [0, 30, 40, 50, 60, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df['age_bin'] = pd.cut(df['years'], bins=bins, labels=labels, include_lowest=True)

    bins_bmi = [0, 18.5, 25, 30, 100]
    labels_bmi = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Class'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, include_lowest=True)

    bins_map = [0, 80, 100, 120, 1000]
    labels_map = ['Low', 'Normal', 'High', 'Very High']
    df['MAP_Class'] = pd.cut(df['MAP'], bins=bins_map, labels=labels_map, include_lowest=True)

    # Clip outliers
    for col in clipping_bounds:
        df[col] = df[col].clip(lower=clipping_bounds[col]['lower'], upper=clipping_bounds[col]['upper'])

    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['MAP'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    df['BMI_Class'] = pd.cut(df['bmi'], bins=bins_bmi, labels=labels_bmi, include_lowest=True)
    df['MAP_Class'] = pd.cut(df['MAP'], bins=bins_map, labels=labels_map, include_lowest=True)

    categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'active', 'age_bin', 'BMI_Class', 'MAP_Class']
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col].astype(str))

    df['Cluster'] = np.nan
    male_idx = df['gender'] == label_encoders['gender'].transform(['2'])[0]
    female_idx = df['gender'] == label_encoders['gender'].transform(['1'])[0]

    cluster_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'active', 'age_bin', 'BMI_Class', 'MAP_Class']
    if km_male and not df[male_idx].empty:
        df.loc[male_idx, 'Cluster'] = km_male.predict(df[male_idx][cluster_cols])
    if km_female and not df[female_idx].empty:
        df.loc[female_idx, 'Cluster'] = km_female.predict(df[female_idx][cluster_cols])

    features = ['years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'active', 'bmi', 'MAP',
                'age_bin', 'BMI_Class', 'MAP_Class', 'Cluster']
    return df[features]
