import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Provided predictions and confidence intervals data
predictions = [99.784, 99.1866, 99.4785, 99.8152, 99.77898, 99.2457,
               99.86968, 99.90826667, 98.9365, 99.861, 99.88111333,
               99.89771333, 99.2501, 99.6783375, 99.5748, 99.76438, 98.56638]
confidence_intervals = pd.DataFrame({
    'lower_bound': [98.710301, 97.404008, 98.068542, 99.167252, 99.214088,
                    97.244239, 99.538291, 99.591798, 97.036945, 99.541803,
                    99.631288, 99.520908, 97.545768, 97.984092, 97.948744,
                    98.483777, 95.914221],
    'upper_bound': [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                    100.0, 100.0, 100.0]
})

# Visualizing the predictions with clipped confidence intervals
plt.figure(figsize=(10, 6))
x_values = range(len(predictions))
plt.plot(x_values, predictions, 'o-', label='Predictions')
plt.fill_between(x_values, confidence_intervals['lower_bound'], confidence_intervals['upper_bound'], color='gray', alpha=0.3, label='Confidence Interval')
plt.xlabel('Test Data Index')
plt.ylabel('Prediction Value')
plt.title('Predictions with Clipped Confidence Intervals')
plt.savefig("/Users/loganvega/Desktop/am_data_driven_work_flow/report/predictions_with_clipped_confidence_intervals.png")
plt.legend()
plt.show()