import streamlit as st

from PIL import Image
import pandas as pd
import numpy as np

import pickle

####  ทำด้านข้าง

with st.sidebar:
    st.title("Contact: ")
    
    st.header("Aphinan Photun")
    
    st.header(" ")
    
    st.write('''

Tel.(+66)932374237

Email: ''', 'mailto:aphinanp62@nu.ac.th'
'''

Address: House No. 6/2 Moo.1, Nong Luang Sub-district, Tha tako District, Nakhon Sawan Province, Thailand 60160

''')
    st.header(" ")
    st.header(" ")
    Im1 = Image.open('Vehicle-Emission-Predictor/Images/nu.png')
    Im2 = Image.open('Vehicle-Emission-Predictor/Images/scinu.png')
    st.image([Im1, Im2] , width= 150)



####  หน้าแรก

st.write("""

# 🚗 Vehicles' Emissions Predictor 💨

Predicting **Emissions** from **Vehicles** using Random Forest.

Air pollution has become a serious issue in large cities and one major source of the pollution is vehicles emissions. 
Vehicles pollutants harm our health and contain greenhouse gases that cause climate change, 
and the most common greenhouse gas is **Carbon Dioxide (CO2)**. 

""")

image = Image.open('Vehicle-Emission-Predictor/Images/Car.jpg')
st.image(image)

st.write("""

Our model use 4 features such as ⚙ Engine Size(L), 🔩 Cylinders, Fuel Consumption(L/100 km) 
for combined city 🌇 and highway 🛣 driving , and ⛽ Fuel Type to predict CO2 emissions(g/km). 
The training and testing accuracy for Vehicles' Emission prediction are 95.57% and 94.41% 
Nonetheless, this application also contains contents for you to understand CO2 emissions.

""")

####  อะไรสักอย่าง




filename = 'Vehicle-Emission-Predictor/model/finalized_model.sav'
model = pickle.load(open(filename, 'rb'))




''''''

st.write('Predict CO2 Emissions')

### Predict CO2 Emissions

left1, right1 = st.beta_columns(2)
with left1: 
    engine = st.number_input(' ⚙ Engine Size(L) (max at 9)' , min_value=1)
with right1:
    fuel_consuption = st.number_input('🌇 Fuel Consumption Combined(L/100 km) 🛣 (max at 35)' , min_value=0)
    
left2, right2 = st.beta_columns(2)
with left2: 
    fuel_type = st.selectbox(
        ' ⛽ Fuel Type ',
        ('Regular Gasoline', 'Premium Gasoline', 'Diesel', 'Ethanol (E85)'))
with right2:
    cylinders = st.slider(' 🔩 Cylinders ', 2, 16, 1)




if st.button(' 🔎 Predict! '):
    
    e = 0
    r = 0
    p = 0
    
    if fuel_type == 'Regular Gasoline':
        r = 1
    elif fuel_type == 'Premium Gasoline':
        p = 1
    elif fuel_type == 'Ethanol (E85)':
        e = 0
    
    X = np.array([[engine, cylinders, fuel_consuption, e, r, p]])
    #st.write(X)
    y = model.predict(X)
    pred = str(round(y.item(0), 2))
    
    st.write('Your Vehicle 🚙 has CO2 Emissions: ***%s*** (g/km)' %(pred))
    
    if y.item(0) > 300:
        st.write('# Your vehicle emits too much pollution.')
    


st.markdown("+ **Python libraries**: streamlit, numpy, pandas, keras, sqlite3, datetime, pickle.")
st.markdown("+ **Data source**: [Open Government of Canada](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64#wb-auto-6)")