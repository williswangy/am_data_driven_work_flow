from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def calculate_accuracy(self, X, y):
        """Calculate and return the MSE and R² for the model on a given dataset."""
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2

    def detect_overfitting(self, X_train, y_train, X_val, y_val):
        """Check for overfitting by comparing training and validation scores."""
        train_mse, train_r2 = self.calculate_accuracy(X_train, y_train)
        val_mse, val_r2 = self.calculate_accuracy(X_val, y_val)

        logger.info(f"Training MSE: {train_mse}, Training R²: {train_r2}")
        logger.info(f"Validation MSE: {val_mse}, Validation R²: {val_r2}")

        if train_mse < val_mse:
            logger.warning("Model may be overfitting to the training data.")

    def generate_report(self, X_train, y_train, X_val, y_val):
        """Generate a report of model performance metrics."""
        train_mse, train_r2 = self.calculate_accuracy(X_train, y_train)
        val_mse, val_r2 = self.calculate_accuracy(X_val, y_val)

        report = (
            f"Model Performance Report:\n"
            f"Training MSE: {train_mse}, Training R²: {train_r2}\n"
            f"Validation MSE: {val_mse}, Validation R²: {val_r2}\n"
        )

        if train_mse < val_mse:
            report += "Warning: Model may be overfitting to the training data."
        else:
            report += "No significant overfitting detected."

        return report
