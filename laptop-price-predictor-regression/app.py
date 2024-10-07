import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# RetinaDisplay
retina_display = st.selectbox('Retina Display',['No','Yes'])

# Clock speed
clock_speed = st.number_input('Speed GHz of the Laptop')

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

# Assuming these are the 14 features your model was trained on
columns = ['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips',
           'RetinaDisplay', 'ppi', 'Cpu brand', 'clockspeed', 'HDD', 'SSD',
           'Gpu brand', 'os']

if st.button('Predict Price'):
    # Convert categorical values
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    if retina_display == 'Yes':
        retina_display = 1
    else:
        retina_display = 0

    # Calculate ppi from resolution and screen size
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    # Create the query as a DataFrame with the correct column names
    query = pd.DataFrame([[company, type, ram, weight, touchscreen, ips, retina_display,
                           ppi, cpu, clock_speed, hdd, ssd, gpu, os]], columns=columns)

    # Ensure the DataFrame is in the correct shape (1, 14)
    st.write(query.shape)  # Should print (1, 14)
    st.write("Query DataFrame:")
    st.write(query)
    st.write("Query DataFrame shape:", query.shape)
    st.write(query.dtypes)
    st.write("scikit-learn version:", sklearn.__version__)

    # Predict using the pipeline and display the result
    prediction = pipe.predict(query)
    st.title(f"Predicted Price: {np.exp(prediction)[0]}")

