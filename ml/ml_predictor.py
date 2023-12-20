import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    """
    A class for loading a trained model and making predictions with it.

    Attributes:
    model_path (str): Path to the saved trained model.
    model: The loaded model object.

    Methods:
    load_model(): Loads the trained model from the specified path.
    predict(X_new): Makes predictions on new data.
    confidence_intervals(X_new, alpha): Calculates confidence intervals for predictions.
    """
    def __init__(self, model_path):
        """
        Initialize the ModelPredictor with the path to the trained model.

        Parameters:
        model_path (str): Path to the trained model file.
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Loads the trained model from the specified path.

        Logs an informational message on successful load or an error message on failure.
        """
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")

    def predict(self, X_new):
        """
        Makes predictions on new data using the loaded model.

        Parameters:
        X_new: New data to make predictions on.

        Returns:
        np.ndarray: An array of predictions if the model is loaded, otherwise None.

        Logs a warning if the model is not loaded and an error if prediction fails.
        """
        if self.model is not None:
            try:
                predictions = self.model.predict(X_new)
                return predictions
            except Exception as e:
                logger.error(f"An error occurred while making predictions: {e}")
        else:
            logger.warning("Model is not loaded. Call load_model() before making predictions.")
            return None

    def confidence_intervals(self, X_new, alpha=0.05):
        """
        Calculates confidence intervals for RandomForest predictions.

        Parameters:
        X_new: New data to make predictions on.
        alpha (float): Significance level for the intervals (default is 0.05).

        Returns:
        pd.DataFrame: A DataFrame with lower and upper bounds of the confidence intervals if successful, otherwise None.

        Logs a warning if the model is not loaded or not a RandomForestRegressor, and an error if calculation fails.
        """
        if self.model is None:
            logger.warning("Model is not loaded. Call load_model() before calculating confidence intervals.")
            return None

        if not isinstance(self.model, RandomForestRegressor):
            logger.warning("This method is specifically tailored for RandomForestRegressor.")
            return None

        try:
            # Extract individual tree predictions from the RandomForest model
            tree_predictions = np.array([tree.predict(X_new) for tree in self.model.estimators_])

            # Calculate mean and standard deviation of tree predictions
            mean_predictions = np.mean(tree_predictions, axis=0)
            std_devs = np.std(tree_predictions, axis=0)

            # Compute the quantiles for the normal distribution
            quantile = stats.norm.ppf(1 - alpha / 2)

            # Calculate the upper and lower bounds of the confidence intervals
            lower_bounds = mean_predictions - quantile * std_devs
            upper_bounds = mean_predictions + quantile * std_devs

            return pd.DataFrame({
                'lower_bound': lower_bounds,
                'upper_bound': upper_bounds
            })
        except Exception as e:
            logger.error(f"An error occurred while calculating confidence intervals: {e}")
            return None
