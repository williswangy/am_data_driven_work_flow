import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path):
        """
        Initialize the Predictor with the path to the trained model.
        :param model_path: Path to the trained model file.
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Loads the trained model from the specified path.
        """
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"An error occurred while loading the model: {e}")

    def predict(self, X_new):
        """
        Makes predictions on new data.
        :param X_new: New data to make predictions on.
        :return: Array of predictions.
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
        :param X_new: New data to make predictions on.
        :param alpha: Significance level for the intervals.
        :return: A DataFrame with lower and upper bounds of the confidence intervals.
        """
        if self.model is None:
            logger.warning("Model is not loaded. Call load_model() before calculating confidence intervals.")
            return None

        if not isinstance(self.model, RandomForestRegressor):
            logger.warning("This method is specifically tailored for RandomForestRegressor.")
            return None

        try:
            # Extract individual tree predictions
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
