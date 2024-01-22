import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data
file_path = 'results.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Separate data for different time points
data_0h = data[data['Timepoint'] == '0h']
data_40h = data[data['Timepoint'] == '40h']

# Define the metrics and colors
metrics = ['integrated visible', '808 abs']
colors = ['#0773B1', '#033560', '#D36027', '#803013']  # Define four colors
# D36027
# Create Matplotlib figure
plt.figure(figsize=(8, 8))

# Define the timepoints
timepoints = [data_0h, data_40h]

# Loop over each combination of timepoint and metric
for i, (data_timepoint, metric, color) in enumerate(zip(timepoints*len(metrics), np.repeat(metrics, len(timepoints)), colors)):
    # Fit a linear regression model to the data
    m, b = np.polyfit(data_timepoint['Loading'], data_timepoint[metric], 1)

    # Calculate the predicted y values
    y_pred = np.polyval([m, b], data_timepoint['Loading'])

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((data_timepoint[metric] - y_pred) ** 2))

    # Print the equation and error of the trendline
    print(f"{metric}, y = {m}x + {b}, RMSE = {rmse}")

    # Add the data points to the plot
    plt.scatter(data_timepoint['Loading'], data_timepoint[metric], label=f'{data_timepoint["Timepoint"].iloc[0]} - {metric}', color=color, s=350, alpha=0.8)

    # Add the trendline to the plot
    plt.plot(data_timepoint['Loading'], y_pred, color=color, linewidth=2)

# Set the title, labels, and other parameters of the plot
# plt.title('Metrics vs Loading', fontsize=24)
fontsize = 40
plt.xlabel('Loading (wt/wt, 1e-4)', fontsize=fontsize)
plt.ylabel('Absorbance', fontsize=fontsize)
plt.xticks(fontsize=fontsize/1.5)
plt.yticks(fontsize=fontsize/1.5, ticks=[0, 1, 2])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tick_params(axis='both', which='major', direction='out', length=6, width=2)
plt.grid(False)
plt.legend(fontsize=fontsize/2)

# Show plot
plt.tight_layout()
plt.show()