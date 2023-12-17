from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the configuration
with open('config.json', 'r') as file:
    config = json.load(file)

# Initialize and use DataPreprocessor
preprocessor = DataPreprocessor(config)
data = preprocessor.load_data()
preprocessor.describe_data()
preprocessor.handle_missing_values()

# Normalize only the specified columns
columns_to_normalize = config['COLUMNS_TO_NORMALIZE']
normalized_data = preprocessor.normalize_data(columns_to_normalize)
preprocessor.describe_data()

# For feature selection, use both the normalized features and the target column
target_column = 'Total density'  # Replace with your actual target column name
features_for_selection = normalized_data[columns_to_normalize + [target_column]]

selector = FeatureSelectorAM(features_for_selection, target_column)
selector.plot_correlation_heatmap(include_target=True)
importance = selector.feature_importance()
logger.info(f"Feature Importance:\n{importance}")
