import streamlit as st
import numpy as np
import joblib

scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

st.title("House price prediction App🏡")

st.divider()

OverallQual = st.number_input("Rate the overall quality of the house from 0-10", value=1, step=1)

GrLivArea = st.number_input("Total above-ground living area", value=500, step=100)

GarageCars = st.number_input("Number of cars that can be accommodated", value=0, step=1)

TotalBsmtSF = st.number_input("Total area of the basement in square feet", value=100, step=100)

YearBuilt = st.number_input("The year in which the house was built", value=1900, step=1)

FullBath = st.number_input("Total number of bathrooms", value=1, step=1)

BedroomAbvGr = st.number_input("Total number of bedrooms", value=1, step=1)

X = [OverallQual,GrLivArea,GarageCars,TotalBsmtSF,YearBuilt,FullBath,BedroomAbvGr]

st.divider()

predict = st.button("Predict!")

st.divider()

if predict:
    X1 = np.array(X)
    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]
    st.success(f"Estimated the house price: {prediction:.2f}")
    st.toast("prediction successful")
    st.snow()
else:
    "Please use the predict button"


