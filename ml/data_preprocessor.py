import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.normalized_data = None


    def load_data(self):
        """Loads data from the given .xlsm filepath."""
        try:
            # Use openpyxl engine to read .xlsm file
            self.data = pd.read_excel(self.filepath, engine='openpyxl')
            logger.info(f"Data loaded successfully from {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.filepath}: {e}")
        return self.data

    def describe_data(self):
        """Provides a description of the dataset."""
        description = str(self.data.describe())
        logger.info("Dataset Statistics:\n" + description)

    def handle_missing_values(self):
        """Checks for and handles missing values in the dataset."""
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found in the dataset:\n{missing_values}")

            # Handling missing values by filling with mean
            self.data.fillna(self.data.mean(), inplace=True)
            logger.info("Missing values have been handled.")
        else:
            logger.info("No missing values found in the dataset.")

    def normalize_data(self, columns):
        """Normalizes specified columns using Min-Max Scaling."""
        try:
            scaler = MinMaxScaler()
            self.data[columns] = scaler.fit_transform(self.data[columns])
            logger.info("Data normalization completed successfully.")
        except Exception as e:
            logger.error(f"Error during data normalization: {e}")
        return self.data
