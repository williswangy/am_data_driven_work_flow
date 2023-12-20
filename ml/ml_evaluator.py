from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    A class responsible for evaluating machine learning models.

    Attributes:
    model_registry (ModelRegistry): An instance of ModelRegistry to retrieve models.

    Methods:
    calculate_accuracy(model_name, X, y): Calculate and return MSE and R² for a model.
    calculate_accuracy_bagging(model_name, X, y): Calculate accuracy for bagging models.
    detect_overfitting(model_name, X_train, y_train, X_val, y_val): Check for overfitting.
    generate_report(model_name, X_train, y_train, X_val, y_val): Generate a performance report.
    evaluate_performance(predictions, y_true): Evaluate performance based on predictions.
    """
    def __init__(self, model_registry):
        """
        Initialize the ModelEvaluator with a model registry.

        Parameters:
        model_registry (ModelRegistry): An instance of ModelRegistry to retrieve models.
        """
        self.model_registry = model_registry

    def calculate_accuracy(self, model_name, X, y):
        """
        Calculate and return the Mean Squared Error (MSE) and R-squared (R²) for a model on a given dataset.

        Parameters:
        model_name (str): The name of the model to evaluate.
        X: Data features for evaluation.
        y: Data labels for evaluation.

        Returns:
        tuple: A tuple containing the MSE and R² values.
        """
        model = self.model_registry.get(model_name)  # Retrieve the specific model
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2

    def calculate_accuracy_bagging(self, model_name, X, y):
        """
        Calculate and return the MSE and R² for bagging models on a given dataset.

        Parameters:
        model_name (str): The name of the bagging model set to evaluate.
        X: Data features for evaluation.
        y: Data labels for evaluation.

        Returns:
        tuple: A tuple containing the MSE and R² values.

        Raises:
        ValueError: If no bagging models are found for the given model name.
        """
        models = self.model_registry.bagging_models.get(model_name, [])
        if not models:
            raise ValueError(f"No bagging models found for {model_name}.")

        all_predictions = np.array([model.predict(X) for model in models])
        avg_predictions = np.mean(all_predictions, axis=0)

        mse = mean_squared_error(y, avg_predictions)
        r2 = r2_score(y, avg_predictions)
        return mse, r2

    def detect_overfitting(self, model_name, X_train, y_train, X_val, y_val):
        """
        Check for overfitting by comparing training and validation scores.

        Parameters:
        model_name (str): The name of the model to check for overfitting.
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.

        Logs the MSE and R² scores for both training and validation data.
        Warns if overfitting is detected.
        """
        train_mse, train_r2 = self.calculate_accuracy(model_name, X_train, y_train)
        val_mse, val_r2 = self.calculate_accuracy(model_name, X_val, y_val)

        logger.info(f"{model_name} Training MSE: {train_mse}, Training R²: {train_r2}")
        logger.info(f"{model_name} Validation MSE: {val_mse}, Validation R²: {val_r2}")

        if train_mse < val_mse:
            logger.warning(f"{model_name} model may be overfitting to the training data.")

    def generate_report(self, model_name, X_train, y_train, X_val, y_val):
        """
        Generate a report of model performance metrics.

        Parameters:
        model_name (str): The name of the model for which to generate a report.
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.

        Returns:
        str: A string report detailing the performance metrics.
        """
        train_mse, train_r2 = self.calculate_accuracy(model_name, X_train, y_train)
        val_mse, val_r2 = self.calculate_accuracy(model_name, X_val, y_val)

        report = (
            f"{model_name} Model Performance Report:\n"
            f"Training MSE: {train_mse}, Training R²: {train_r2}\n"
            f"Validation MSE: {val_mse}, Validation R²: {val_r2}\n"
        )

        if train_mse < val_mse:
            report += "Warning: Model may be overfitting to the training data."
        else:
            report += "No significant overfitting detected."

        return report

    def evaluate_performance(self, predictions, y_true):
        """
        Evaluate the performance of a model given its predictions and the true values.

        Parameters:
        predictions: The predicted values from the model.
        y_true: The actual true values.

        Returns:
        tuple: A tuple containing the MSE and R² values.

        Logs the performance and returns MSE and R² metrics.
        Handles exceptions by logging them and returning None for both metrics.
        """
        try:
            mse = mean_squared_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            logger.info(f"Performance - MSE: {mse}, R²: {r2}")
            return mse, r2
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return None, None
