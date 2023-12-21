import json
import logging
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
from ml_trainer import ModelTrainer, ModelRegistry
from ml_evaluator import ModelEvaluator
from ml_predictor import ModelPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
import warnings
from sklearn.model_selection import GridSearchCV

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

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_train_test_split(config['TEST_SIZE'],
                                                                                          config["VALIDATION_SIZE"],
                                                                                          config['RANDOM_STATE'])
    trainer = ModelTrainer(model_registry)
    evaluator = ModelEvaluator(model_registry)

    for model_name in model_registry.models.keys():
        trainer.train_model(model_name, X_train, y_train)
        #trainer.validate_model(model_name, X_train, y_train, cv_method='kfold', cv_folds=5)
        model_report = evaluator.generate_report(model_name, X_train, y_train, X_val, y_val)
        logger.info(f"{model_name} Model Performance Report:\n" + model_report)

    return model_registry, X_test, y_test


def train_and_evaluate_random_forest(preprocessor, config):
    """
    Trains and evaluates the RandomForest model with hyperparameter tuning and cross-validation.

    Parameters:
    preprocessor (DataPreprocessor): An instance of the DataPreprocessor class for data preparation.
    config (dict): Configuration settings for the RandomForest model.

    Returns:
    RandomForestRegressor: The trained RandomForest model.
    """
    model_registry = ModelRegistry()

    # Initialize RandomForest with default parameters
    random_forest = RandomForestRegressor(random_state=42)

    # Register the model
    model_registry.register("RandomForest", random_forest)

    # Prepare training and validation data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_train_test_split(config['TEST_SIZE'],
                                                                                          config["VALIDATION_SIZE"],
                                                                                          config['RANDOM_STATE'])

    # Initialize trainer and evaluator
    trainer = ModelTrainer(model_registry)
    evaluator = ModelEvaluator(model_registry)

    # Hyperparameter tuning (example using GridSearchCV or a similar approach)
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Retrieve the best parameters and log them
    best_params = grid_search.best_params_
    logger.info(f"Best Hyperparameters for RandomForest: {best_params}")

    # Retrain the model with the best parameters
    tuned_random_forest = RandomForestRegressor(**best_params, random_state=42)
    model_registry.register("TunedRandomForest", tuned_random_forest)

    # Train and evaluate using cross-validation
    trainer.validate_model("TunedRandomForest", X_train, y_train, cv_method='kfold', cv_folds=5)

    # Optional: Evaluate on the validation set if needed
    tuned_random_forest.fit(X_train, y_train)
    model_report = evaluator.generate_report("TunedRandomForest", X_train, y_train, X_val, y_val)
    logger.info(f"TunedRandomForest Model Performance Report:\n" + model_report)
    #trainer.save_model("TunedRandomForest", config['MODEL_SAVE_PATH'])

    return tuned_random_forest, X_test, y_test



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

    model_registry, X_test, y_test =train_and_evaluate_random_forest(preprocessor, config)

    # #model_registry, X_test, y_test = train_and_evaluate_models(preprocessor, config)
    predictions, confidence_intervals = make_predictions(model_registry, X_test, y_test, config)

    logger.info(f"Predictions:\n{predictions}\nConfidence Intervals:\n{confidence_intervals}")


if __name__ == "__main__":
    main('config.json')
