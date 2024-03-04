from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from clases_y_funciones import CustomStandardScaler, Load_Keras_Model, pipe_regression, pipe_classification, X_train
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import os

#Creo una función que convierte el input del usuario en una fila del dataset
#utilizado para entrenar el modelo, de esta forma se podrá usar para hacer predicciones

def apply_feat_eng_to_user_input(user_input):
    '''Aplica la ingeniería de características al input del usuario'''
    user_input_processed = user_input.copy()  # Copia el DataFrame para no modificar el original
    
    direccion_a_angulo = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }

    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        angulo = direccion_a_angulo.get(user_input[col].values[0], 0)
        user_input_processed[col+'_x'] = np.cos(np.radians(angulo))
        user_input_processed[col+'_y'] = np.sin(np.radians(angulo))
    
    # Eliminar las columnas originales de las direcciones de viento
    user_input_processed.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1, inplace=True)

    # Convertir los meses a coordenadas cartesianas
    user_input_processed['Month_X'] = np.cos(2 * np.pi * user_input['Month'] / 12)
    user_input_processed['Month_Y'] = np.sin(2 * np.pi * user_input['Month'] / 12)
    user_input_processed.drop(columns=['Month'], axis=1, inplace=True)
    return user_input_processed

input_features = ['MinTemp', 'MaxTemp','Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'Month']

categorical_features = ['WindGustDir','WindDir9am','WindDir3pm','Month', 'RainToday']

def get_user_input():

    input_dict = {}

    with st.form(key='my_form'):
        for feat in input_features:
            if feat in categorical_features:
                input_value = st.selectbox(f'Introduzca valor para {feat}', X_train[feat].unique())
                input_dict[feat] = input_value
            else:
                input_value = st.number_input(f"Introduzca valor para {feat}", value=0.0, step=0.01)
                input_dict[feat] = input_value


        submit_button = st.form_submit_button(label='Submit')
    input_df = pd.DataFrame([input_dict])
    input_df.to_csv('input_df.csv', index=False)
    return input_df, submit_button


user_input, submit_button = get_user_input()

user_input_processed = apply_feat_eng_to_user_input(user_input)

# Cuando se aprete submit se realiza la predicción
if submit_button:
    #Se realiza la prediccion de si llovera mañana
    prediction = pipe_classification.predict(user_input_processed)
    prediction_classification_value = prediction[0]
    llueve_mañana = prediction_classification_value > 0.5

    # Se muestra el resultado
    st.header("Predicción de lluvia ¿mañana llueve?")

    if llueve_mañana:
        estado = 'Mañana llueve'
    else:
        estado = 'Mañana no llueve'
    st.write(estado)

    #Si mañana llueve se calcula cuanto
    if llueve_mañana:
        st.header("Cantidad de lluvia")
        prediction = pipe_regression.predict(user_input_processed)
        prediction_regression_value = prediction[0]
        st.write(prediction_regression_value)
    

