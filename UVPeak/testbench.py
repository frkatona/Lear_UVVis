import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import to_rgb, to_hex
from colorsys import rgb_to_hls, hls_to_rgb
import os

def read_spectrum_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Wavelength', 'Absorbance'])
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    # df = df[(df['Wavelength'] >= 300) & (df['Wavelength'] <= 850)]
    return df

def smooth_data(df, sigma=2):
    df_smoothed = df.copy()
    df_smoothed['Absorbance'] = gaussian_filter1d(df['Absorbance'], sigma=sigma)
    return df_smoothed

def adjust_saturation(hex_color, saturation_factor):
    rgb = to_rgb(hex_color)
    h, l, s = rgb_to_hls(*rgb)
    new_s = min(1, s * saturation_factor)
    new_rgb = hls_to_rgb(h, l, new_s)
    return to_hex(new_rgb)

# Paths to the data directory and colors file
data_dir = r'UVPeak\Spectra\230829_cbPDMS_wtLoadings_again'  # Replace with the actual path
colors_file = r'UVPeak\Colorlists\colorlist_lear.txt'  # Replace with the actual path

with open(colors_file, 'r') as file:
    hex_colors = [line.strip() for line in file if line.strip()]

files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
files = sorted(files, key=lambda f: (f.split('_')[0], float(f.split('_')[1]), float(f.split('_')[2])))
files = files[0:6]
loadings = sorted(set(f.split('_')[1] for f in files))
color_mappings = {loading: hex_colors[i % len(hex_colors)] for i, loading in enumerate(loadings)}

plt.figure(figsize=(15, 10))

for file in files:
    loading = file.split('_')[1]
    df = read_spectrum_data(os.path.join(data_dir, file))
    smoothed_df = smooth_data(df)
    plt.plot(smoothed_df['Wavelength'], smoothed_df['Absorbance'], color=color_mappings[loading], label=f'{file}')

plt.axvline(x=400, color='purple', linestyle='--', linewidth=3)
plt.axvline(x=700, color='red', linestyle='--', linewidth=3)
plt.axvline(x=808, color='black', linestyle='--', linewidth=3)
plt.xlabel('Wavelength')
plt.ylabel('Absorbance')
plt.title('Full UV-Vis Spectra with Colors Based on Loadings')
plt.legend()
plt.show()