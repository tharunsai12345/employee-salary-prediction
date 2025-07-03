%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

@st.cache_resource
def load_model():
    with gzip.open("income_classifier_model.pkl.gz", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("income_scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

st.title("💼 Employee Income Prediction App")
st.markdown("Predict whether an employee earns `>50K` or `<=50K` per year.")

with st.form("income_form"):
    age = st.number_input("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    fnlwgt = st.number_input("fnlwgt", 10000, 1000000, 100000)
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Some-college', '11th', 'Masters', 'Assoc-acdm', 'Doctorate', '10th', '9th'])
    educational_num = st.slider("Education Number", 1, 16, 9)
    marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing'])
    relationship = st.selectbox("Relationship", ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Other'])
    gender = st.radio("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada'])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
        'educational-num': educational_num, 'marital-status': marital_status,
        'occupation': occupation, 'relationship': relationship, 'race': race,
        'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
        'hours-per-week': hours_per_week, 'native-country': native_country
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Income Group: **{prediction}**")
