# import json
# import logging
# from data_preprocessor import DataPreprocessor
# from feature_selector import FeatureSelectorAM
# from ml_trainer import ModelTrainer, ModelRegistry
# from ml_evaluator import ModelEvaluator
# from ml_predictor import ModelPredictor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression, Lasso
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load configuration
# config_path = 'config.json'
# with open(config_path, 'r') as file:
#     config = json.load(file)
#
# # Initialize and use DataPreprocessor
# preprocessor = DataPreprocessor(config)
# data = preprocessor.load_data()
#
# preprocessor.plot_data_distribution(save_path=config['FREQUENCY_PLOT_SAVE_PATH'], nrows=3, ncols=3)
# preprocessor.describe_data()
# preprocessor.handle_missing_values()
#
# # Normalize specified columns
# columns_to_normalize = config['COLUMNS_TO_NORMALIZE']
# normalized_data = preprocessor.normalize_data(columns_to_normalize)
# if normalized_data is not None:
#     logger.info("Normalized Data Statistics:\n" + str(normalized_data.describe()))
# else:
#     logger.error("Normalization failed. Cannot proceed with train-test split.")
#
# # Initialize FeatureSelectorAM
# selector = FeatureSelectorAM(normalized_data, columns_to_normalize, config['TARGET_COLUMN'])
# selector.plot_correlation_heatmap(include_target=True, save_path=config['HEATMAP_SAVE_PATH'])
# importance = selector.feature_importance()
# logger.info(f"Feature Importance:\n{importance}")
#
#
# # Initialize ModelRegistry and register models
# model_registry = ModelRegistry()
# models_to_use = {
#     'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'LinearRegression': LinearRegression(),
#     'Lasso': Lasso(alpha=0.1)
# }
#
# for model_name, model in models_to_use.items():
#     model_registry.register(model_name, model)
#
# # Train-test split
# X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_train_test_split(config['TEST_SIZE'],
#                                                                                       config["VALIDATION_SIZE"],
#                                                                                       config['RANDOM_STATE'])
# y_train_transformed = np.log1p(y_train)
# y_val_transformed = np.log1p(y_val)
# # Train, validate, and evaluate models
# trainer = ModelTrainer(model_registry)
# evaluator = ModelEvaluator(model_registry)
#
# for model_name in model_registry.models.keys():
#     trainer.train_model(model_name, X_train, y_train)
#     # For k-fold cross-validation
#     #trainer.validate_model(model_name, X_train, y_train, cv_method='kfold', cv_folds=5)
#     #trainer.validate_model(model_name, X_train, y_train, cv_method='loocv')
#     model_report = evaluator.generate_report(model_name,X_train, y_train, X_val, y_val)
#     logger.info(f"{model_name} Model Performance Report:\n" + model_report)
#
#     # if model_name == 'RandomForest':
#     #     trainer.save_model(model_name, config['MODEL_SAVE_PATH'])
#
#
# predictor = ModelPredictor(config['MODEL_SAVE_PATH'])
# predictor.load_model()
# predictions = predictor.predict(X_test)
#
# evaluator.evaluate_performance(predictions,y_test)
#
# confidence_intervals = predictor.confidence_intervals(X_test)
# print(confidence_intervals)

import json
import logging
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
from ml_trainer import ModelTrainer, ModelRegistry
from ml_evaluator import ModelEvaluator
from ml_predictor import ModelPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration settings from a JSON file.

    Parameters:
    config_path (str): Path to the configuration file.

    Returns:
    dict: Configuration settings.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def process_data(config):
    """
    Processes the dataset based on the configuration settings.

    Parameters:
    config (dict): Configuration settings for data preprocessing.

    Returns:
    DataPreprocessor: An instance of the DataPreprocessor class after loading and processing data.
    """
    preprocessor = DataPreprocessor(config)
    data = preprocessor.load_data()
    preprocessor.describe_data()
    preprocessor.handle_missing_values()
    return preprocessor

def select_features(preprocessor, config):
    """
    Selects features from the dataset using normalization and correlation analysis.

    Parameters:
    preprocessor (DataPreprocessor): An instance of the DataPreprocessor class.
    config (dict): Configuration settings for feature selection.

    Returns:
    FeatureSelectorAM: An instance of the FeatureSelectorAM class after feature selection.
    """
    columns_to_normalize = config['COLUMNS_TO_NORMALIZE']
    normalized_data = preprocessor.normalize_data(columns_to_normalize)
    if normalized_data is None:
        logger.error("Normalization failed. Cannot proceed with feature selection.")
        return None

    selector = FeatureSelectorAM(normalized_data, columns_to_normalize, config['TARGET_COLUMN'])
    selector.plot_correlation_heatmap(include_target=True, save_path=config['HEATMAP_SAVE_PATH'])
    importance = selector.feature_importance()
    logger.info(f"Feature Importance:\n{importance}")
    return selector

def train_and_evaluate_models(preprocessor, config):
    """
    Trains and evaluates models specified in the configuration.

    Parameters:
    preprocessor (DataPreprocessor): An instance of the DataPreprocessor class.
    config (dict): Configuration settings for model training and evaluation.

    Returns:
    tuple: A tuple containing the ModelRegistry instance, test features, and test labels.
    """
    model_registry = ModelRegistry()
    models_to_use = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1)
    }

    for model_name, model in models_to_use.items():
        model_registry.register(model_name, model)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_train_test_split(config['TEST_SIZE'], config["VALIDATION_SIZE"], config['RANDOM_STATE'])
    trainer = ModelTrainer(model_registry)
    evaluator = ModelEvaluator(model_registry)

    for model_name in model_registry.models.keys():
        trainer.train_model(model_name, X_train, y_train)
        model_report = evaluator.generate_report(model_name, X_train, y_train, X_val, y_val)
        logger.info(f"{model_name} Model Performance Report:\n" + model_report)

    return model_registry, X_test, y_test

def make_predictions(model_registry, X_test, y_test, config):
    """
    Makes predictions on the test data and evaluates the model's performance.

    Parameters:
    model_registry (ModelRegistry): An instance of the ModelRegistry class.
    X_test: Test data features.
    y_test: Test data labels.
    config (dict): Configuration settings for making predictions.

    Returns:
    tuple: A tuple containing the predictions and confidence intervals.
    """
    predictor = ModelPredictor(config['MODEL_SAVE_PATH'])
    predictor.load_model()
    predictions = predictor.predict(X_test)
    evaluator = ModelEvaluator(model_registry)
    evaluator.evaluate_performance(predictions, y_test)
    confidence_intervals = predictor.confidence_intervals(X_test)
    logger.info(f"Confidence Intervals:\n{confidence_intervals}")
    return predictions, confidence_intervals


def main(config_path):
    """
    Main function to run the complete machine learning pipeline.

    Parameters:
    config_path (str): Path to the configuration file.
    """
    config = load_config(config_path)

    preprocessor = process_data(config)
    if preprocessor is None:
        return

    selector = select_features(preprocessor, config)
    if selector is None:
        return

    model_registry, X_test, y_test = train_and_evaluate_models(preprocessor, config)
    predictions, confidence_intervals = make_predictions(model_registry, X_test, y_test, config)

    logger.info(f"Predictions:\n{predictions}\nConfidence Intervals:\n{confidence_intervals}")


if __name__ == "__main__":
    main('config.json')
