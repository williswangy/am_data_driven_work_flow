import json
import logging
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
from ml_trainer import ModelTrainer, ModelRegistry
from ml_evaluator import ModelEvaluator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = 'config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

# Initialize and use DataPreprocessor
preprocessor = DataPreprocessor(config)
data = preprocessor.load_data()

preprocessor.plot_data_distribution(save_path=config['FREQUENCY_PLOT_SAVE_PATH'], nrows=3, ncols=3)
preprocessor.describe_data()
preprocessor.handle_missing_values()

# Normalize specified columns
columns_to_normalize = config['COLUMNS_TO_NORMALIZE']
normalized_data = preprocessor.normalize_data(columns_to_normalize)
if normalized_data is not None:
    logger.info("Normalized Data Statistics:\n" + str(normalized_data.describe()))
else:
    logger.error("Normalization failed. Cannot proceed with train-test split.")

# Initialize FeatureSelectorAM
selector = FeatureSelectorAM(normalized_data, columns_to_normalize, config['TARGET_COLUMN'])
selector.plot_correlation_heatmap(include_target=True, save_path=config['HEATMAP_SAVE_PATH'])
importance = selector.feature_importance()
logger.info(f"Feature Importance:\n{importance}")

# Initialize ModelRegistry and register models
model_registry = ModelRegistry()
models_to_use = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(alpha=0.1)
}

for model_name, model in models_to_use.items():
    model_registry.register(model_name, model)

# Train-test split
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.create_train_test_split(config['TEST_SIZE'],
                                                                                      config["VALIDATION_SIZE"],
                                                                                      config['RANDOM_STATE'])

# Train, validate, and evaluate models
trainer = ModelTrainer(model_registry)
evaluator = ModelEvaluator(model_registry)

for model_name in model_registry.models.keys():
    trainer.train_model(model_name, X_train, y_train)
    trainer.validate_model(model_name, X_train, y_train, cv_folds=5)
    model_report = evaluator.generate_report(model_name,X_train, y_train, X_val, y_val)
    logger.info(f"{model_name} Model Performance Report:\n" + model_report)

    # Train and evaluate models with bagging
    # trainer.train_model_with_bagging(model_name, X_train, y_train, n_bootstrap=10)
    # bagged_report = evaluator.generate_report_bagging(X_train, y_train, X_test, y_test)
    # logger.info(f"{model_name} Bagging Model Performance Report:\n" + bagged_report)
