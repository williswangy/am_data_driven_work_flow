import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class for preprocessing data including loading, handling missing values, normalization, and splitting.

    Attributes:
    config (dict): Configuration settings containing file paths and other parameters.
    data (pd.DataFrame): Raw data loaded from the file.
    normalized_data (pd.DataFrame): Data after applying normalization.

    Methods:
    load_data(): Loads data from a configured file path.
    describe_data(): Outputs a statistical description of the dataset.
    handle_missing_values(): Handles missing values in the dataset.
    normalize_data(columns): Normalizes specified columns in the dataset.
    create_train_test_split(test_size, validation_size, random_state): Splits data into training, validation, and test sets.
    plot_data_distribution(save_path, nrows, ncols): Plots the distribution of each feature and saves the figure.
    """
    def __init__(self, config):
        """
        Initialize the DataPreprocessor with configuration settings.

        Parameters:
        config (dict): Configuration settings including file paths and target column names.
        """
        self.config = config
        self.data = None
        self.normalized_data = None

    def load_data(self):
        """
        Loads data from the configured .xlsm filepath using pandas.

        Returns:
        pd.DataFrame: The loaded data.

        Logs an informational message on successful load or an error message on failure.
        """
        try:
            self.data = pd.read_excel(self.config['DATA_FILE_PATH'], engine='openpyxl')
            if 'Parameter' in self.data.columns:
                self.data.drop('Parameter', axis=1, inplace=True)
                logger.info("'Parameter' column dropped from the dataset.")
            logger.info(f"Data loaded successfully from {self.config['DATA_FILE_PATH']}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.config['DATA_FILE_PATH']}: {e}")
        return self.data

    def describe_data(self):
        """
        Provides a statistical description of the dataset including mean, standard deviation, etc.

        Logs the description to the logger.
        """
        description = str(self.data.describe())
        logger.info("Dataset Statistics:\n" + description)

    def handle_missing_values(self):
        """
        Checks for and handles missing values in the dataset by filling them with the mean.

        Logs the presence of missing values and a message after handling them.
        """
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found in the dataset:\n{missing_values}")
            self.data.fillna(self.data.mean(), inplace=True)
            logger.info("Missing values have been handled.")
        else:
            logger.info("No missing values found in the dataset.")

    def normalize_data(self, columns):
        """
        Normalizes specified columns in the dataset using Min-Max Scaling.

        Parameters:
        columns (list): List of columns to normalize.

        Returns:
        pd.DataFrame: The DataFrame with normalized data.

        Logs an informational message on successful normalization or an error message on failure.
        """
        try:
            scaler = MinMaxScaler()
            normalized_columns = scaler.fit_transform(self.data[columns])
            self.normalized_data = self.data.copy()
            self.normalized_data[columns] = normalized_columns
            logger.info("Data normalization completed successfully.")
            return self.normalized_data
        except Exception as e:
            logger.error(f"Error during data normalization: {e}")
            return None

    def create_train_test_split(self, test_size=0.2, validation_size=0.1, random_state=None):
        """
        Splits the data into training, validation, and testing sets.

        Parameters:
        test_size (float): Proportion of the dataset to include in the test split.
        validation_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Controls the shuffling applied to the data before splitting.

        Returns:
        tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test).

        Logs an informational message on successful split.
        """
        if self.normalized_data is not None:
            X = self.normalized_data.drop(self.config['TARGET_COLUMN'], axis=1)
            y = self.normalized_data[self.config['TARGET_COLUMN']]
        else:
            logger.warning("Data has not been normalized. Using original data for split.")
            X = self.data.drop(self.config['TARGET_COLUMN'], axis=1)
            y = self.data[self.config['TARGET_COLUMN']]

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        validation_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=validation_size_adjusted, random_state=random_state)

        logger.info("Train-validation-test split completed successfully.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def plot_data_distribution(self, save_path, nrows=3, ncols=3):
        """
        Plots the distribution of each feature in the dataset and saves the figure.

        Parameters:
        save_path (str): Path where the plot will be saved.
        nrows (int): Number of rows for subplots.
        ncols (int): Number of columns for subplots.

        Logs an error message if there is no data available for plotting.
        """
        if self.data is None:
            logger.error("No data available to plot.")
            return

        num_plots = len(self.data.columns)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))
        axes = axes.flatten()

        for i, column in enumerate(self.data.columns):
            sns.histplot(self.data[column], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {column}")
            axes[i].set_xlabel(column)
            axes[i].set_ylabel("Frequency")

        for j in range(i + 1, nrows * ncols):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved combined distribution plot to {save_path}")
