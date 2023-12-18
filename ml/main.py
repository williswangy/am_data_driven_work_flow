import json
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelectorAM
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

if normalized_data is not None:
    logger.info("Normalized Data Statistics:\n" + str(normalized_data.describe()))


# Initialize FeatureSelectorAM with the specified feature columns
selector = FeatureSelectorAM(normalized_data, columns_to_normalize, config['TARGET_COLUMN'])
selector.plot_correlation_heatmap(include_target=True, save_path=config['HEATMAP_SAVE_PATH'])
importance = selector.feature_importance()
logger.info(f"Feature Importance:\n{importance}")