import streamlit as st
import numpy as np
import pandas as pd
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
from ml_trainer import ModelTrainer, ModelRegistry
from ml_evaluator import ModelEvaluator
from ml_predictor import ModelPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

# Configuration and Logging (simplified for Streamlit)
import json

config_path = 'config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

# Initialize components (assuming these are callable as is)
preprocessor = DataPreprocessor(config)
model_registry = ModelRegistry()
predictor = ModelPredictor(config['MODEL_SAVE_PATH'])


# Streamlit App
def main():
    st.title("Data Science Pipeline")

    # Sample input for prediction (you may want to create dynamic input fields based on your features)
    st.sidebar.header("Input Features for Prediction")
    input_feature1 = st.sidebar.number_input("Input Feature 1", min_value=0.0, max_value=100.0, value=50.0)
    input_feature2 = st.sidebar.number_input("Input Feature 2", min_value=0.0, max_value=100.0, value=50.0)

    # Button to perform prediction
    if st.sidebar.button("Predict"):
        # Prepare data for prediction
        input_data = pd.DataFrame([[input_feature1, input_feature2]], columns=['feature1', 'feature2'])
        predictor.load_model()
        predictions = predictor.predict(input_data)
        st.write(f"Predictions: {predictions}")


if __name__ == '__main__':
    main()
