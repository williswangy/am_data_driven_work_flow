from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model_registry):
        self.model_registry = model_registry

    def calculate_accuracy(self, model_name, X, y):
        """Calculate and return the MSE and R² for a model on a given dataset."""
        model = self.model_registry.get(model_name)  # Retrieve the specific model
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2

    def calculate_accuracy_bagging(self, model_name, X, y):
        """Calculate and return the MSE and R² for bagging models on a given dataset."""
        models = self.model_registry.bagging_models.get(model_name, [])
        if not models:
            raise ValueError(f"No bagging models found for {model_name}. Ensure the model is trained with bagging.")

        all_predictions = np.array([model.predict(X) for model in models])
        avg_predictions = np.mean(all_predictions, axis=0)

        mse = mean_squared_error(y, avg_predictions)
        r2 = r2_score(y, avg_predictions)
        return mse, r2

    def detect_overfitting(self, model_name, X_train, y_train, X_val, y_val):
        """Check for overfitting by comparing training and validation scores."""
        train_mse, train_r2 = self.calculate_accuracy(model_name, X_train, y_train)
        val_mse, val_r2 = self.calculate_accuracy(model_name, X_val, y_val)

        logger.info(f"{model_name} Training MSE: {train_mse}, Training R²: {train_r2}")
        logger.info(f"{model_name} Validation MSE: {val_mse}, Validation R²: {val_r2}")

        if train_mse < val_mse:
            logger.warning(f"{model_name} model may be overfitting to the training data.")

    def generate_report(self, model_name, X_train, y_train, X_val, y_val):
        """Generate a report of model performance metrics."""
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
        """Evaluate model performance given predictions and true values."""
        try:
            # Calculate MSE and R² using the predictions and true values
            mse = mean_squared_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)

            # Log the performance
            logger.info(f"Performance - MSE: {mse}, R²: {r2}")

            # Return the performance metrics
            return mse, r2
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return None, None

