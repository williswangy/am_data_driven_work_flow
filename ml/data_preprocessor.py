import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.normalized_data = None


    def load_data(self):
        """Loads data from the given .xlsm filepath."""
        try:
            # Use openpyxl engine to read .xlsm file
            self.data = pd.read_excel(self.config['DATA_FILE_PATH'], engine='openpyxl')
            # Check if 'Parameter' column exists in the data and drop it
            if 'Parameter' in self.data.columns:
                self.data.drop('Parameter', axis=1, inplace=True)
                logger.info("'Parameter' column dropped from the dataset.")
            logger.info(f"Data loaded successfully from {self.config['DATA_FILE_PATH']}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.config['DATA_FILE_PATH']}: {e}")
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
        """Normalizes specified columns using Min-Max Scaling, while keeping the target column."""
        try:
            scaler = MinMaxScaler()
            # Normalize only specified columns
            normalized_columns = scaler.fit_transform(self.data[columns])

            # Create a new DataFrame for normalized data
            self.normalized_data = self.data.copy()
            self.normalized_data[columns] = normalized_columns

            logger.info("Data normalization completed successfully.")

            return self.normalized_data

        except Exception as e:
            logger.error(f"Error during data normalization: {e}")
            return None

    def create_train_test_split(self, test_size=0.2, random_state=None):
        """
        Splits the data into training and testing sets.
        """
        if self.normalized_data is not None:
            X = self.normalized_data.drop(self.config['TARGET_COLUMN'], axis=1)
            y = self.normalized_data[self.config['TARGET_COLUMN']]
        else:
            logger.warning("Data has not been normalized. Using original data for split.")
            X = self.data.drop(self.config['TARGET_COLUMN'], axis=1)
            y = self.data[self.config['TARGET_COLUMN']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info("Train-test split completed successfully.")
        return X_train, X_test, y_train, y_test




