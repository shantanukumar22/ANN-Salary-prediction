import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing  import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model("regression_model.h5")

with open("le_gender.pkl","rb") as file:
    le_gender=pickle.load(file)
with open("ohe_geo.pkl","rb") as file:
    ohe_geo=pickle.load(file)
with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

st.title("Customer salary prediction using ANN")


geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1])
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

input_data = pd.DataFrame(
    {
        'CreditScore':[credit_score],
        'Gender':[le_gender.transform([gender])[0]],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'Exited':[exited],
    }
)


 
# One Hot Encode 'Geography'
geo_encoded = ohe_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
 
# Cobine the one-hot encoded columns with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=True)
 
# Scale the data
input_data_scaled = scaler.transform(input_data)
 
# Prediction of estimated salary
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]
 
st.write(f'Predected estimated salary: {predicted_salary: .2f}')