import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf


## loading models
model = load_model('model.h5')

#loading encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender=pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo=pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler=pickle.load(f)

## streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to churn.")
#input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0].tolist())
gender = st.selectbox("Gender", label_encoder_gender.classes_.tolist())
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure=st.slider("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([input_data, geo_encoded_df], axis=1)

#scaling
input_scaled=scaler.transform(input_df)

#prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    print(prediction)
    if prediction[0][0] > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")