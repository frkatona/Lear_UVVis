import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import trapz

def plot_uv_vis_data(directory_path):
    # Define colormaps for the timepoints
    colormaps = {
        '0h': cm.Oranges,
        '40h': cm.Blues,
        'oven': cm.Greens
    }

    # Define the new order for the loadings
    loading_order = ['0', '1e-6', '1e-5', '1e-4', '1e-3']

    # Load the data from the directory
    dataframes = []
    loadings = []
    timepoints = []

    for file in os.listdir(directory_path):
        if file.endswith(".txt"):
            parts = file.split('_')
            loading = parts[1]
            timepoint = parts[2]
            
            filepath = os.path.join(directory_path, file)
            df = pd.read_csv(filepath, sep="\\t", skiprows=15, names=['Wavelength', 'Absorbance'])
            df['Smoothed Absorbance'] = savgol_filter(df['Absorbance'], 51, 3)  # Smooth the data
            
            dataframes.append(df)
            loadings.append(loading)
            timepoints.append(timepoint)

    # Sort data by timepoints and then by the new loadings order for legend ordering
    sorted_indices = sorted(range(len(timepoints)), key=lambda k: (timepoints[k], loading_order.index(loadings[k])))
    sorted_dataframes = [dataframes[i] for i in sorted_indices] 
    sorted_loadings = [loadings[i] for i in sorted_indices]
    sorted_timepoints = [timepoints[i] for i in sorted_indices]

    # Compute normalized AUC for each sample in the 400-700 nm range
    auc_values = []
    normalization_factors = {tp: 0 for tp in set(sorted_timepoints)}
    for df, loading, timepoint in zip(sorted_dataframes, sorted_loadings, sorted_timepoints):
        subset_df = df[(df['Wavelength'] >= 400) & (df['Wavelength'] <= 700)]
        auc = trapz(subset_df['Smoothed Absorbance'], subset_df['Wavelength'])
        if loading == "0":
            normalization_factors[timepoint] = auc
        auc_values.append(auc)

    normalized_auc_values = [auc / normalization_factors[tp] for auc, tp in zip(auc_values, sorted_timepoints)]
    
    # Create the combined plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    # Line graph on the left side
    for df, loading, timepoint in zip(sorted_dataframes, sorted_loadings, sorted_timepoints):
        if loading in ['1e-4', '1e-3']:
            alpha_value = 0.05
        else:
            alpha_value = 1.0
        
        color = colormaps[timepoint](np.linspace(0.25, 1, len(loading_order))[loading_order.index(loading)])
        ax[0].plot(df['Wavelength'], df['Smoothed Absorbance'], color=color, linewidth=2, alpha=alpha_value, label=f'Loading: {loading}, Time: {timepoint}')

    ax[0].set_xlabel('Wavelength (nm)', fontsize=16)
    ax[0].set_ylabel('Abs', fontsize=16)
    ax[0].set_xlim(400, 700)
    ax[0].set_ylim(0, 0.6)
    ax[0].tick_params(axis='both', which='major', labelsize=14, direction='out')
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Bar graph on the right side
    unique_timepoints = list(set(sorted_timepoints))
    barWidth = 0.4
    r1 = np.arange(len(loading_order))
    r2 = [x + barWidth for x in r1]
    bars1_values = [normalized_auc_values[i] for i, tp in enumerate(sorted_timepoints) if tp == unique_timepoints[0]]
    bars2_values = [normalized_auc_values[i] for i, tp in enumerate(sorted_timepoints) if tp == unique_timepoints[1]]
    
    ax[1].bar(r1, bars1_values, color=colormaps[unique_timepoints[0]](0.75), width=barWidth, edgecolor='white', label=unique_timepoints[0], alpha=0.7)
    ax[1].bar(r2, bars2_values, color=colormaps[unique_timepoints[1]](0.75), width=barWidth, edgecolor='white', label=unique_timepoints[1], alpha=0.7)
    ax[1].set_xticks([r + barWidth for r in range(len(loading_order))])
    ax[1].set_xticklabels(loading_order, fontsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=14, direction='out')
    ax[1].legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Call the function with the path to your directory
directory_path = r"UVPeak\\UV-Vis_CB-loading_Thesis-Revisions\\ambient"
plot_uv_vis_data(directory_path)