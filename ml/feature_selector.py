import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier depending on your target variable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FeatureSelectorAM:
    def __init__(self, data, target_column):
        self.data = data
        self.target = data[target_column]
        self.features = data.drop(target_column, axis=1)
        self.logger = logging.getLogger(__name__)

    def plot_correlation_heatmap(self,include_target=False):
        try:
            data_for_heatmap = self.features
            if include_target:
                data_for_heatmap = pd.concat([self.features, self.target], axis=1)

            plt.figure(figsize=(12, 10))
            correlation_matrix = data_for_heatmap.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Feature Correlation Heatmap")
            plt.show()
            self.logger.info("Correlation heatmap plotted successfully.")
        except Exception as e:
            self.logger.error(f"Failed to plot correlation heatmap: {e}")

    def feature_importance(self, n_estimators=100):
        try:
            model = RandomForestRegressor(n_estimators=n_estimators)
            model.fit(self.features, self.target)
            importance = pd.Series(model.feature_importances_, index=self.features.columns)
            self.logger.info("Feature importance calculated successfully.")
            return importance.sort_values(ascending=False)
        except Exception as e:
            self.logger.error(f"Failed to calculate feature importance: {e}")