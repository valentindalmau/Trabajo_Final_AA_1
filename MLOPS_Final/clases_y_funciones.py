# clases y funciones del pipeline de preprocesamiento.
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split

data = pd.read_csv("df2.csv")

X = data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday', 'Month']]

y = data['RainfallTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1),test_size=0.2,random_state = 42, shuffle=False)

def feat_eng(X_train, X_test, y_train, y_test):
    '''Ingeniería de características antes de rellenar valores faltantes'''
    # Convertir las direcciones de viento a coordenadas cartesianas
    direccion_a_angulo = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        X_train[col+'_angulo'] = X_train[col].map(direccion_a_angulo)
        X_train[col+'_x'] = np.cos(np.radians(X_train[col+'_angulo']))
        X_train[col+'_y'] = np.sin(np.radians(X_train[col+'_angulo']))
        X_test[col+'_angulo'] = X_test[col].map(direccion_a_angulo)
        X_test[col+'_x'] = np.cos(np.radians(X_test[col+'_angulo']))
        X_test[col+'_y'] = np.sin(np.radians(X_test[col+'_angulo']))

    # Eliminar las columnas originales de las direcciones de viento y de los ángulos correspondientes
    columns_to_drop = [col+'_angulo' for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']]
    X_train.drop(columns=columns_to_drop, axis=1, inplace=True)
    X_test.drop(columns=columns_to_drop, axis=1, inplace=True)
    

    # Convertir los meses a coordenadas cartesianas
    X_train['Month_X'] = np.cos(2 * np.pi * X_train['Month'] / 12)
    X_train['Month_Y'] = np.sin(2 * np.pi * X_train['Month'] / 12)
    X_test['Month_X'] = np.cos(2 * np.pi * X_test['Month'] / 12)
    X_test['Month_Y'] = np.sin(2 * np.pi * X_test['Month'] / 12)

    #Codificar RainToday
    X_train['RainToday'] = X_train['RainToday'].replace({'Yes': 1, 'No': 0})
    X_test['RainToday'] = X_test['RainToday'].replace({'Yes': 1, 'No': 0})

    # Creo los imputadores
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Separo variables numericas de categóricas para cada tipo de imputacion
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    # Imputacion por media y moda evitando data leakage
    X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_test[numeric_features] = numeric_imputer.transform(X_test[numeric_features])
    X_test[categorical_features] = categorical_imputer.transform(X_test[categorical_features])

    return X_train, X_test, y_train, y_test



class CustomStandardScaler(BaseEstimator, TransformerMixin):
    '''
    Estandarización z-score
    '''
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self, X_train, y=None):
        
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns

        # Inicilizar
        self.scaler = StandardScaler()

        # Fittear
        self.scaler.fit(X_train[numeric_features])

        return self

    def transform(self, X):
        # Identificar numericas
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

        # Transform only numeric features
        X_numeric = X[numeric_features]
        X_transformed = self.scaler.transform(X_numeric)
        X_numeric_scaled = pd.DataFrame(X_transformed, columns=X_numeric.columns)

        # Concatenar numericas con categoricas
        X_result = pd.concat([X_numeric_scaled, X.select_dtypes(exclude='number')], axis=1)

        return X_result
    
class Load_Keras_Model():
    def __init__(self, filepath):
        self.load_model(filepath)
        self.filepath = filepath
        
    def fit(self, X=None, y=None):
        self.load_model(self.filepath)
        return self

    def predict(self, X):
        input_df = pd.DataFrame(X)
        input_df.to_csv('debug_df.csv', index=False)
        if self.model == None:
          raise ValueError('Model not loaded')
        predictions = self.model.predict(X)
        return predictions

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

# Se crea el pipeline de regresion y clasificacion con los modelos exportados
pipe_regression = Pipeline([
    ('zscore_scaler', CustomStandardScaler()), 
    ('neural_network', Load_Keras_Model(filepath='best_model_regression.h5'))
])
pipe_classification = Pipeline([
    ('zscore_scaler', CustomStandardScaler()),
    ('neural_network', Load_Keras_Model(filepath='best_model_classification.h5'))
])

#Realizo la ingeniería de características, imputacion y estandarizacion
X_train, X_test, y_train, y_test = feat_eng(X_train, X_test, y_train, y_test)

#Dropeo solo para el fiteo de las pipes las columnas que no usé en el colab, de esta forma
#Se pueden usar para que el usuario pase el input pero no se usaran para predecir

pipe_regression.fit(X_train.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Month']), y_train)
pipe_classification.fit(X_train.drop(columns=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Month']), y_train)

