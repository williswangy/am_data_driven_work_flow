from sklearn.model_selection import cross_val_score, LeaveOneOut
import joblib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    A class used to register and retrieve machine learning models.

    Attributes:
    models (dict): A dictionary to store the models with their names as keys.

    Methods:
    register(model_name, model): Registers a new model under the given name.
    get(model_name): Retrieves a model by its name.
    """
    def __init__(self):
        """Initialize the ModelRegistry with an empty dictionary for models."""
        self.models = {}

    def register(self, model_name, model):
        """
        Register a new model.

        Parameters:
        model_name (str): The name under which the model is to be registered.
        model: The machine learning model to be registered.
        """
        self.models[model_name] = model

    def get(self, model_name):
        """
        Retrieve a model by its name.

        Parameters:
        model_name (str): The name of the model to retrieve.

        Returns:
        The requested model if it exists, otherwise raises a ValueError.

        Raises:
        ValueError: If the model_name is not found in the registry.
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' not registered.")

class ModelTrainer:
    """
    A class responsible for training, validating, and saving machine learning models.

    Attributes:
    model_registry (ModelRegistry): An instance of ModelRegistry to retrieve models.
    bagging_models (dict): A dictionary to store bagging models (not used in current implementation).

    Methods:
    train_model(model_name, X_train, y_train): Trains a model with provided data.
    validate_model(model_name, X, y, cv_method, cv_folds): Validates a model using cross-validation.
    save_model(model_name, file_path): Saves a model to a file.
    """
    def __init__(self, model_registry):
        """
        Initialize the ModelTrainer with a model registry.

        Parameters:
        model_registry (ModelRegistry): An instance of ModelRegistry to retrieve models.
        """
        self.model_registry = model_registry
        self.bagging_models = {}

    def train_model(self, model_name, X_train, y_train):
        """
        Train a model with the provided training data.

        Parameters:
        model_name (str): The name of the model to train.
        X_train: Training data features.
        y_train: Training data labels.
        """
        try:
            model = self.model_registry.get(model_name)
            model.fit(X_train, y_train)
            logger.info(f"{model_name} model training completed.")
        except Exception as e:
            logger.error(f"Error during {model_name} model training: {e}")

    def validate_model(self, model_name, X, y, cv_method='kfold', cv_folds=5):
        """
        Validate a model using cross-validation.

        Parameters:
        model_name (str): The name of the model to validate.
        X: Data features for validation.
        y: Data labels for validation.
        cv_method (str): The method of cross-validation ('kfold' or 'loocv').
        cv_folds (int): Number of folds for k-fold cross-validation.

        Outputs cross-validation mean squared error scores to the logger.
        """
        try:
            model = self.model_registry.get(model_name)
            if cv_method == 'loocv':
                cv = LeaveOneOut()
            else:
                cv = cv_folds
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            mse_scores = -scores
            logger.info(f"{model_name} Cross-Validation MSE Scores: {mse_scores}")
            logger.info(f"{model_name} Average MSE Score: {np.mean(mse_scores)}")
        except Exception as e:
            logger.error(f"Error during {model_name} model validation: {e}")

    def save_model(self, model_name, file_path):
        """
        Save a trained model to a file.

        Parameters:
        model_name (str): The name of the model to be saved.
        file_path (str): The path where the model should be saved.
        """
        try:
            model = self.model_registry.get(model_name)
            joblib.dump(model, file_path)
            logger.info(f"{model_name} model saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving the {model_name} model: {e}")
