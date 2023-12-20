import joblib
import numpy as np
import pandas as pd
from scipy import stats

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
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

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
                print(f"An error occurred while making predictions: {e}")
        else:
            print("Model is not loaded. Call load_model() before making predictions.")
            return None

    def confidence_intervals(self, X_new, alpha=0.05):
        """
        Calculates confidence intervals for predictions.
        :param X_new: New data to make predictions on.
        :param alpha: Significance level for the intervals.
        :return: A DataFrame with lower and upper bounds of the confidence intervals.
        """
        if self.model is None:
            print("Model is not loaded. Call load_model() before calculating confidence intervals.")
            return None

        # Assuming the model has a `predict` method that can return standard deviations of predictions
        # This will be model-specific; not all models provide standard deviations of predictions
        try:
            predictions = self.model.predict(X_new)
            if hasattr(self.model, 'predict_std'):
                std_devs = self.model.predict_std(X_new)
            else:
                # If the model does not support prediction intervals, we need a different approach
                print("The model does not support prediction standard deviations for confidence intervals.")
                return None

            intervals = stats.norm.interval(1 - alpha, loc=predictions, scale=std_devs)
            return pd.DataFrame({
                'lower_bound': intervals[0],
                'upper_bound': intervals[1]
            })
        except Exception as e:
            print(f"An error occurred while calculating confidence intervals: {e}")
            return None

# Example usage:
# predictor = Predictor('path_to_trained_model.pkl')
# predictor.load_model()
# new_data = pd.DataFrame(...)  # Your new data here
# predictions = predictor.predict(new_data)
# confidence_intervals = predictor.confidence_intervals(new_data)
