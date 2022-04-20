import streamlit as st
import os
from PIL import Image


def app():
    st.title('Introduction')
    image = Image.open(os.path.join('pages', 'resources', 'welcome.jpg'))
    st.image(image, caption='Air pollution')

    st.subheader('About')

    ## FALTA O CHECK ON GITHUB
    st.write("""
    This application provides an overview of the brazilian_houses_to_rent dataset from Kaggle. It is a dataset that provides rent prices for real estate properties in Brazil.
    The data were provided from this [source](https://www.kaggle.com/rubenssjr/brasilian-houses-to-rent). 
    You can check on the sidebar:
    - EDA (Exploratory Data Analysis)
    - Model Prediction
    - Model Evaluation
    The prediction are made regarding to the rent amount utilizing pre trained machine learning models.
    All the operations in the dataset were already done and stored as csv files inside the data directory. If you want to check the code, go through the notebook directory in the [github repository](https://github.com/arturlunardi/predict_rental_prices_streamlit).
    """)

    st.subheader('Model Definition')

    st.write("""
    The structure of the training it is to wrap the process around a scikit-learn Pipeline. There were 4 possible combinations and 5 models, resulting in 20 trained models.
    The combinations are regarding to perform Feature Creation and/or Target Transformations in the dataset.
    Models:
    - Random Forest
    - XGB
    - Ridge
    - LGBM
    - Neural Network
    Our main accuracy metric is RMSE. To enhance our model definition, we utilized Cross Validation and Random Search for hyperparameter tuning.
    """)
