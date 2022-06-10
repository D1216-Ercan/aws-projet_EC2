import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
#from xgboost import sklearn 



st.title("LET US CALCULATE YOUR CAR'S PRICE")
img = Image.open("car_photo.jpg")
st.image(img, caption="Calculating in the most accurate way", width=300)
st.sidebar.markdown("### Please enter your vehicle's informations")
make_model=st.sidebar.selectbox("Select your vehicle's model", ['Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa','Opel Insignia', 'Renault Clio', 'Renault Duster','Renault Espace'])
st.write("You selected this model:", make_model)
body_type=st.sidebar.selectbox("Select your vehicle's model", ['Sedans', 'Station wagon', 'Compact', 'Coupe', 'Van', 'Off-Road',
       'Convertible', 'Transporter'])
st.write("You selected this body type:", body_type)
km = st.sidebar.number_input("Select your vehicle's km", min_value=0, max_value=317000, value=0, step=100)
st.write("Your vehicle's km:", km)
age = st.sidebar.slider("Select your vehicle's age", min_value=0, max_value=3, value=0, step=1)
st.write("Your vehicle's age:", age)
Gearing_Type = st.sidebar.selectbox("Your Vehicle's Gearing Type", ['Automatic', 'Manual', 'Semi-automatic'])
st.write("You selected this Gearing_Type:", Gearing_Type)
Fuel=st.sidebar.selectbox("Your Vehicle's Fuel Type", ['Diesel', 'Benzine', 'LPG/CNG', 'Electric'])
st.write("You selected this Fuel Type:", Fuel)
hp_kW = st.sidebar.number_input("Select your Vehicle's HP", min_value=40, max_value=300, value=40, step=10)
st.write("Your vehicle's HP:", hp_kW)
import pickle
filename = 'auto_scout.pkl'
model = pickle.load(open(filename, 'rb'))
enc = pickle.load(open("autoscout_encoder.pkl", "rb"))
car = {
    "make_model": [make_model],
    "body_type" : [body_type],
    "km": [km],
    "age": [age],
    "Gearing_Type": [Gearing_Type],
    "Fuel": [Fuel],
    "hp_kW": [hp_kW]
}
df=pd.DataFrame(car)
cat = df.select_dtypes("object").columns
df[cat] = enc.transform(df[cat])
c1, c2, c3, c4, c5, c6 ,c7 ,c8 ,c9 = st.columns(9)
if c5.button('Predict'):
    result = model.predict(df)[0]
    st.info(f"Predicted value of your car : {round(result)}$") 