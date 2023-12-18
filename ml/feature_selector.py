import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelectorAM:
    def __init__(self, data, feature_columns, target_column):
        self.data = data
        self.target = data[target_column]
        self.features = data[feature_columns]

    def plot_correlation_heatmap(self, include_target=False, save_path=None):
        """
        Generates and saves a heatmap showing the correlation between features.
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
        try:
            model = RandomForestRegressor(n_estimators=n_estimators)
            model.fit(self.features, self.target)
            importance = pd.Series(model.feature_importances_, index=self.features.columns)
            logger.info("Feature importance calculated successfully.")
            return importance.sort_values(ascending=False)
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")