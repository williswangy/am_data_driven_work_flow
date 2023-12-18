from sklearn.model_selection import cross_val_score
import joblib
import logging
import numpy as np

from sklearn.base import clone
from sklearn.utils import resample


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
        self.bagging_models = {}

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

    def train_model_with_bagging(self, model_name, X_train, y_train, n_bootstrap=10):
        try:
            model = self.model_registry.get(model_name)
            bagging_models = []

            for i in range(n_bootstrap):
                # Creating bootstrap sample
                X_bootstrap, y_bootstrap = resample(X_train, y_train)

                # Cloning the model to train on each bootstrap separately
                cloned_model = clone(model)
                cloned_model.fit(X_bootstrap, y_bootstrap)
                bagging_models.append(cloned_model)

            self.bagging_models[model_name] = bagging_models
            logger.info(f"{model_name} model training with bagging completed. {n_bootstrap} models trained.")

        except Exception as e:
            logger.error(f"Error during {model_name} model training with bagging: {e}")

    def predict_with_bagging(self, model_name, X_test):
        try:
            models = self.bagging_models.get(model_name, [])
            if not models:
                raise ValueError(f"No bagging models found for {model_name}. Ensure the model is trained with bagging.")

            predictions = [model.predict(X_test) for model in models]
            avg_prediction = np.mean(predictions, axis=0)
            return avg_prediction

        except Exception as e:
            logger.error(f"Error during prediction with {model_name} bagging models: {e}")
            return None




