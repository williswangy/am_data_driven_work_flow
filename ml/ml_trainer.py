from sklearn.model_selection import cross_val_score
import joblib
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, model_name, model):
        self.models[model_name] = model

    def get(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not registered.")

class ModelTrainer:
    def __init__(self, model_registry):
        self.model_registry = model_registry

    def train_model(self, model_name, X_train, y_train):
        try:
            model = self.model_registry.get(model_name)
            model.fit(X_train, y_train)
            logger.info(f"{model_name} model training completed.")
        except Exception as e:
            logger.error(f"Error during {model_name} model training: {e}")

    def validate_model(self, model_name, X_val, y_val, cv_folds=5):
        try:
            model = self.model_registry.get(model_name)
            scores = cross_val_score(model, X_val, y_val, cv=cv_folds, scoring='neg_mean_squared_error')
            mse_scores = -scores
            logger.info(f"{model_name} Cross-Validation MSE Scores: {mse_scores}")
            logger.info(f"{model_name} Average MSE Score: {np.mean(mse_scores)}")
        except Exception as e:
            logger.error(f"Error during {model_name} model validation: {e}")

    def save_model(self, model_name, file_path):
        try:
            model = self.model_registry.get(model_name)
            joblib.dump(model, file_path)
            logger.info(f"{model_name} model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving the {model_name} model: {e}")




