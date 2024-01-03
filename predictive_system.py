import numpy as np
import pickle
import streamlit as st
import pandas as pd

# loading the saved model
with open('trained_model.sav', 'rb') as model_files:
    loaded_model = pickle.load(open(r'E:\Breast Cancer\trained_model.sav', 'rb'))

st.title('Breast Cancer Prediction')

# Get input features from the user

input_features = []

# Assuming i have a list of features
feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']
for feature_name in feature_names:
    value =  st.text_input(f"Enter the value of {feature_name}:")
    input_features.append(value)

# adding a prediction button
    
if st.button('Predict'):
    #Convert input festures to a NumPy array
    input_features_array = np.array(input_features).reshape(1,-1)

    #make prediction using loaded model
    predicted_class = loaded_model.prdict(input_features_array)

    st.write(f"Prediction: {predicted_class[0]}")
    
