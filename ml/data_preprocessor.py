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

    def describe_and_handle_missing_values(self):
        """Checks for and handles missing values in the dataset."""
        logger.info("Checking and handling missing values.")

        # Print initial dataset statistics
        logger.info("Initial Dataset Statistics:\n" + str(self.data.describe()))

        # Check for missing values
        if self.data.isnull().values.any():
            logger.warning("Missing values found in the dataset.")

            # Handling missing values
            self.data.fillna(self.data.mean(), inplace=True)

            logger.info("Missing values have been handled.")
            logger.info("Dataset statistics after handling missing values:\n" + str(self.data.describe()))
        else:
            logger.info("No missing values found in the dataset.")

    def normalize_data(self, columns):
        """Normalizes specified columns using Min-Max Scaling."""
        try:
            scaler = MinMaxScaler()
            self.data[columns] = scaler.fit_transform(self.data[columns])

            # Convert the description to a string using `str()` and log it
            normalized_description = str(self.data[columns].describe())
            logger.info(
                "Data normalization completed successfully. Normalized data statistics:\n" + normalized_description)
        except Exception as e:
            logger.error(f"Error during data normalization: {e}")
        return self.data
