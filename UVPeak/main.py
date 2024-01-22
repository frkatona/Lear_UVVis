import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import to_rgb, to_hex
from colorsys import rgb_to_hls, hls_to_rgb
import os

# Function to read and smooth spectrum data
def read_spectrum_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['Wavelength', 'Absorbance'])
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    return df

def smooth_data(df, sigma=2):
    df_smoothed = df.copy()
    df_smoothed['Absorbance'] = gaussian_filter1d(df['Absorbance'], sigma=sigma)
    return df_smoothed

# Function to adjust saturation of a color
def adjust_saturation(hex_color, saturation_factor):
    rgb = to_rgb(hex_color)
    h, l, s = rgb_to_hls(*rgb)
    new_s = min(1, s * saturation_factor)
    new_rgb = hls_to_rgb(h, l, new_s)
    return to_hex(new_rgb)

# Functions for spectrum analysis
def integrate_spectrum(df, wavelength_range=(400, 700)):
    df_filtered = df[(df['Wavelength'] >= wavelength_range[0]) & (df['Wavelength'] <= wavelength_range[1])]
    return np.trapz(df_filtered['Absorbance'], df_filtered['Wavelength'])

def get_absorbance_at_wavelength(df, wavelength=808):
    closest_wavelength = df.iloc[(df['Wavelength'] - wavelength).abs().argsort()[:1]]
    return closest_wavelength['Absorbance'].values[0]

# Paths to the data directory and colors file
data_dir = r'UVPeak\Spectra\230829_cbPDMS_wtLoadings_again'
colors_file = r'UVPeak\Colorlists\colorlist_lear.txt'  # Replace with the actual path

# Reading the colors
with open(colors_file, 'r') as file:
    # hex_colors = [line.strip() for line in file if line.strip()]
    hex_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF', '#00FFFF']

# Processing the data files
files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
categories = sorted(set('_'.join(f.split('_')[:2]) for f in files), key=lambda x: (x.split('_')[0], float(x.split('_')[1])))
print("categories:")
print(categories)
categories = categories[0:4]

color_mappings = {}
integrated_values = {}
absorbance_at_808 = {}

for category in categories:
    files_in_category = [f for f in files if f.startswith(category)]
    if len(files_in_category) > 1:
        color_mappings[category] = [adjust_saturation(hex_colors[i % len(hex_colors)], 0.5 + 0.5 * (i / (len(files_in_category) - 1))) for i in range(len(files_in_category))]
    else:
        color_mappings[category] = [adjust_saturation(hex_colors[0], 1.0)]  # Full saturation if only one file

    integrated_values[category] = []
    absorbance_at_808[category] = []
    for i, file in enumerate(files_in_category):
        df = read_spectrum_data(os.path.join(data_dir, file))
        smoothed_df = smooth_data(df)
        integrated_values[category].append(integrate_spectrum(smoothed_df))
        absorbance_at_808[category].append(get_absorbance_at_wavelength(smoothed_df))

# Plotting the spectra
plt.figure(figsize=(15, 10))
for category in categories:
    for i, file in enumerate(files_in_category):
        df = read_spectrum_data(os.path.join(data_dir, file))
        smoothed_df = smooth_data(df)
        plt.plot(smoothed_df['Wavelength'], smoothed_df['Absorbance'], color=color_mappings[category][i], label=f'{category}, Time {i}')
plt.axvline(x=400, color='grey', linestyle='--', linewidth=1)
plt.axvline(x=700, color='grey', linestyle='--', linewidth=1)
plt.xlim(300, 850)

fontsize = 40
plt.xlabel('Wavelength /cm$^{-1}$', fontsize=fontsize)
plt.ylabel('Absorbance', fontsize=fontsize)
# plt.xticks(fontsize=fontsize/2)
# plt.yticks(fontsize=fontsize/2, ticks=[0, 1, 2])
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tick_params(axis='both', which='major', direction='out', length=6, width=2)
plt.grid(False)
plt.legend(fontsize=fontsize/2)

plt.show()

# # Plotting the bar graphs
# fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# # Bar plot for integrated values
# for category, values in integrated_values.items():
#     ax[0].bar([f'{category}, Time {i}' for i in range(len(values))], values, color=color_mappings[category])
# ax[0].set_title('Integrated Spectrum Values for Each Spectrum')
# ax[0].set_ylabel('Integrated Absorbance')
# ax[0].set_xlabel('Category and Time')
# ax[0].tick_params(axis='x', rotation=45)

# # Bar plot for absorbance at 808 nm
# for category, values in absorbance_at_808.items():
#     ax[1].bar([f'{category}, Time {i}' for i in range(len(values))], values, color=color_mappings[category])
# ax[1].set_title('Absorbance at 808 nm for Each Spectrum')
# ax[1].set_ylabel('Absorbance')
# ax[1].set_xlabel('Category and Time')
# ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()