import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re

# Function to extract loading and time from the file name
def extract_loading_time(file_name):
    match = re.search(r'ambient_([0-9e\-]+)_([0-9]+)_Absorbance', file_name)
    if match:
        loading = float(match.group(1))
        time = int(match.group(2))
        return loading, time
    return None, None

# Function to read the spectral data from a file
def read_spectral_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Find the line number where the spectral data begins
        for i, line in enumerate(lines):
            if '>>>>>Begin Spectral Data<<<<<' in line:
                start_line = i + 1
                break
        # Read the spectral data into a DataFrame
        data = pd.DataFrame([x.split('\t') for x in lines[start_line:]], columns=['Wavelength', 'Absorbance'])
        data = data.astype(float)
    return data

# Function to plot the spectral data
def plot_spectral_data(data_directory):
    # Define colors for 0h and 40h
    colors = {0: '#0773B1', 40: '#D36027'}
    # Define the new labels and corresponding opacities for the loadings
    loading_labels = {0: '0', 1e-6: '1e-6', 1e-5: '1e-5', 1e-4: '1e-4', 1e-3: '1e-3'}
    opacity_values = {0: 0.5, 1e-6: 0.75, 1e-5: 0.85, 1e-4: 0.1, 1e-3: 0}

    # Initialize a dictionary to store the data
    spectral_data = {}

    # Process each file in the directory
    for file_name in os.listdir(data_directory):
        loading, time = extract_loading_time(file_name)
        if loading is not None and time is not None:
            file_path = os.path.join(data_directory, file_name)
            data = read_spectral_data(file_path)
            spectral_data[(loading, time)] = data

    # Plotting the data
    plt.figure(figsize=(16, 10))
    for (loading, time), data in spectral_data.items():
        # Use the specified opacity for the loading
        opacity = opacity_values[loading]
        # Adjust the color based on the time and the specified opacity
        base_color = colors[time]
        adjusted_color = mcolors.to_rgba(base_color, alpha=opacity)

        # Plot the spectrum with the new label
        label = f'Loading: {loading_labels[loading]}, Time: {time}h'
        plt.plot(data['Wavelength'], data['Absorbance'], label=label, color=adjusted_color)

    plt.axvline(x=400, color='grey', linestyle='--', linewidth=2)
    plt.axvline(x=700, color='grey', linestyle='--', linewidth=2)
    plt.axvline(x=808, color='grey', linestyle='--', linewidth=2)

    fontSize = 40
    plt.grid(False)
    plt.xlim(300, 825)
    plt.ylim(0, 0.5)
    plt.xticks(fontsize=fontSize/2)
    plt.yticks(fontsize=fontSize/2, ticks=[0, 0.1, .2, .3, .4, 0.5])
    plt.tick_params(direction='out')
    plt.xlabel('Wavelength (nm)', fontsize=fontSize)
    plt.ylabel('Absorbance', fontsize=fontSize)
    # plt.legend()
    plt.show()

# Replace this with the path to your data directory
data_directory = r'UVPeak\Spectra\230829_cbPDMS_wtLoadings_again'
plot_spectral_data(data_directory)