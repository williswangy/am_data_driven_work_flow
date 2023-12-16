from data_preprocessor import DataPreprocessor


data_file_path = '/Users/loganvega/Desktop/am_data_driven_work_flow/data/AssessmentData.xlsx'

# Instantiate the class with the file path
preprocessor = DataPreprocessor(data_file_path)

# Load the data
data = preprocessor.load_data()

# Describe the data and handle missing values
preprocessor.describe_and_handle_missing_values()

normalized_data = preprocessor.normalize_data(['Laser Power', 'Hatch Distance', 'X_coord', 'Y_coord', 'Gas flow rate in m/s', 'Laser Speed'])