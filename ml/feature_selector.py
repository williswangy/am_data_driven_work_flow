import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
import os

from sklearn.ensemble import RandomForestRegressor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelectorAM:
    """
    A class for feature selection in a dataset using correlation analysis and recursive feature elimination.

    Attributes:
    data (pd.DataFrame): The entire dataset.
    target (pd.Series): The target column from the dataset.
    features (pd.DataFrame): The feature columns from the dataset.

    Methods:
    plot_correlation_heatmap(include_target, save_path): Generates and saves a heatmap of feature correlations.
    feature_importance(n_estimators): Calculates feature importance using RandomForestRegressor.
    recursive_feature_elimination(estimator, n_features_to_select): Performs RFE to select top features.
    """
    def __init__(self, data, feature_columns, target_column):
        """
        Initializes the FeatureSelectorAM with data, feature columns, and the target column.

        Parameters:
        data (pd.DataFrame): The dataset containing both features and target.
        feature_columns (list): List of column names representing features.
        target_column (str): Name of the target column.
        """
        self.data = data
        self.target = data[target_column]
        self.features = data[feature_columns]

    def plot_correlation_heatmap(self, include_target=False, save_path=None):
        """
        Generates and optionally saves a heatmap showing the correlation between features.

        Parameters:
        include_target (bool): Whether to include the target column in the heatmap.
        save_path (str): Path to save the heatmap image.

        Logs an informational message on successful save or an error message on failure.
        """
        try:
            data_for_heatmap = self.features
            if include_target:
                data_for_heatmap = pd.concat([self.features, self.target], axis=1)

            plt.figure(figsize=(12, 10))
            correlation_matrix = data_for_heatmap.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Feature Correlation Heatmap")

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Correlation heatmap saved successfully at {save_path}")

            plt.show()
        except Exception as e:
            logger.error(f"Failed to plot and/or save correlation heatmap: {e}")

    def feature_importance(self, n_estimators=100):
        """
        Calculates and returns the feature importance using RandomForestRegressor.

        Parameters:
        n_estimators (int): The number of trees in the forest.

        Returns:
        pd.Series: A Series with feature importances, sorted in descending order.

        Logs an informational message on successful calculation or an error message on failure.
        """
        try:
            model = RandomForestRegressor(n_estimators=n_estimators)
            model.fit(self.features, self.target)
            importance = pd.Series(model.feature_importances_, index=self.features.columns)
            logger.info("Feature importance calculated successfully.")
            return importance.sort_values(ascending=False)
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")

    def recursive_feature_elimination(self, estimator, n_features_to_select=5):
        """
        Performs Recursive Feature Elimination (RFE) and returns selected top features.

        Parameters:
        estimator: The base estimator to fit on the reduced set of features.
        n_features_to_select (int): The number of features to select.

        Returns:
        np.array: An array of selected feature names.

        Logs an informational message on successful completion or an error message on failure.
        """
        try:
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
            rfe.fit(self.features, self.target)
            selected_features = self.features.columns[rfe.support_]
            logger.info("Recursive feature elimination completed successfully.")
            return selected_features
        except Exception as e:
            logger.error(f"Failed to perform recursive feature elimination: {e}")
