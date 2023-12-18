import json
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
from ml_trainer import ModelTrainer,ModelRegistry
from ml_evaluator import ModelEvaluator

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import logging

from sklearn.svm import SVC

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the configuration
with open('config.json', 'r') as file:
    config = json.load(file)

# Initialize ModelRegistry and register models
model_registry = ModelRegistry()
model_registry.register('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42))
model_registry.register('LinearRegression', LinearRegression())

# Initialize and use DataPreprocessor
preprocessor = DataPreprocessor(config)
data = preprocessor.load_data()
preprocessor.describe_data()
preprocessor.handle_missing_values()

# Normalize only the specified columns
columns_to_normalize = config['COLUMNS_TO_NORMALIZE']
normalized_data = preprocessor.normalize_data(columns_to_normalize)

if normalized_data is not None:
    logger.info("Normalized Data Statistics:\n" + str(normalized_data.describe()))

else:
    logger.error("Normalization failed. Cannot proceed with train-test split.")


# Initialize FeatureSelectorAM with the specified feature columns
selector = FeatureSelectorAM(normalized_data, columns_to_normalize, config['TARGET_COLUMN'])
selector.plot_correlation_heatmap(include_target=True, save_path=config['HEATMAP_SAVE_PATH'])
importance = selector.feature_importance()
logger.info(f"Feature Importance:\n{importance}")

X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(config['TEST_SIZE'], config['RANDOM_STATE'])

# Use ModelTrainer with registered models
trainer = ModelTrainer(model_registry)

# Train and validate models
trainer.train_model('RandomForest', X_train, y_train)
trainer.validate_model('RandomForest', X_test, y_test, cv_folds=5)
trainer.train_model('LinearRegression', X_train, y_train)
trainer.validate_model('LinearRegression', X_test, y_test, cv_folds=5)


# Initialize and use ModelEvaluator for performance reports
rf_evaluator = ModelEvaluator(model_registry.get('RandomForest'))
rf_report = rf_evaluator.generate_report(X_train, y_train, X_test, y_test)
logger.info("Random Forest Model Performance Report:\n" + rf_report)

lr_evaluator = ModelEvaluator(model_registry.get('LinearRegression'))
lr_report = lr_evaluator.generate_report(X_train, y_train, X_test, y_test)
logger.info("Linear Regression Model Performance Report:\n" + lr_report)





