import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV data into a DataFrame
df = pd.read_csv('results.csv')
color_0h = '#f16e1c'
color_40h = '#4191c6'


# Group the dataframe by 'Loading' and aggregate the results
grouped_df = df.groupby('Loading').agg(list).reset_index()

# Create new arrays for the bar plots considering the grouped data
r_grouped = np.arange(len(grouped_df))
barWidth_wide = 0.8

# Function to format the x-axis labels
def format_loading_label(x):
    if x == 0.0:
        return '0'
    elif x == 0.0001:
        return '1e-04'
    else:
        return str(x)

# Create a figure and axes for the adjusted 'Normalized AUC' plot
fig1_adj, ax1_adj = plt.subplots(figsize=(12, 6))

# Adjusted bar plot for 'Normalized AUC'
for i, loading in enumerate(grouped_df['Loading']):
    ax1_adj.bar(i - barWidth_wide/4, grouped_df['Normalized AUC'][i][0], color=color_0h, width=barWidth_wide/2, edgecolor='white', alpha=0.7)
    ax1_adj.bar(i + barWidth_wide/4, grouped_df['Normalized AUC'][i][1], color=color_40h, width=barWidth_wide/2, edgecolor='white', alpha=0.7)

# Adjusted x ticks and labels for the 'Normalized AUC' plot
ax1_adj.set_xticks(r_grouped)
ax1_adj.set_xticklabels([format_loading_label(x) for x in grouped_df['Loading']], fontsize=16)

# Removing grid lines and setting tick parameters
ax1_adj.grid(False)
ax1_adj.tick_params(axis='both', which='major', labelsize=14, direction='out')

# Setting x-axis title
ax1_adj.set_xlabel('CB loading (wt/wt)', fontsize=16)

# Adjusted legend for the 'Normalized AUC' plot
ax1_adj.legend(['0h', '40h'], loc='upper left', fontsize=12)

# Create a figure and axes for the adjusted '808 absorbance' plot
fig2_adj, ax2_adj = plt.subplots(figsize=(12, 6))

# Adjusted bar plot for '808 absorbance'
for i, loading in enumerate(grouped_df['Loading']):
    ax2_adj.bar(i - barWidth_wide/4, grouped_df['808 absorbance'][i][0], color=color_0h, width=barWidth_wide/2, edgecolor='white', alpha=0.7)
    ax2_adj.bar(i + barWidth_wide/4, grouped_df['808 absorbance'][i][1], color=color_40h, width=barWidth_wide/2, edgecolor='white', alpha=0.7)

# Adjusted x ticks and labels for the '808 absorbance' plot
ax2_adj.set_xticks(r_grouped)
ax2_adj.set_xticklabels([format_loading_label(x) for x in grouped_df['Loading']], fontsize=16)

# Removing grid lines and setting tick parameters
ax2_adj.grid(False)
ax2_adj.tick_params(axis='both', which='major', labelsize=14, direction='out')

# Setting x-axis title
ax2_adj.set_xlabel('CB loading (wt/wt)', fontsize=16)

# Adjusted legend for the '808 absorbance' plot
ax2_adj.legend(['0h', '40h'], loc='upper left', fontsize=12)

# Display the plots
plt.show()