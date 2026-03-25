import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the output directory
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

# Helper function to load and clean data (-999 is the placeholder for missing values)
def load_clean(filename):
    df = pd.read_csv(filename, sep='\t')
    df = df.replace(-999, np.nan)
    return df

# Your custom color scheme mapped to the Species
# B = S. balticum, F = S. fuscum, M = S. majus
colors = {'B': '#1E90FF', 'F': '#FFA500', 'M': '#55CAFF'} 

# Load datasets
try:
    df1 = load_clean('Dataset 1_Precipitation_dependence.dat')
    df2 = load_clean('Dataset 2_Carbon_fluxes.dat')
    df3 = load_clean('Dataset 3_VWCvsNEE.dat')
    df4 = load_clean('Dataset 4_VWCvsPSII.dat')
    df5 = load_clean('Dataset 5_Recovery.dat')
except FileNotFoundError as e:
    print(f"Error loading files. Ensure the .dat files are in the same folder as this script. Details: {e}")
    exit()

print("Generating visualizations...")

# ---------------------------------------------------------
# Plot 1: Precipitation Dependence (FP) by Water Table Depth
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
# Filter to just dry and moist conditions
df1_clean = df1.dropna(subset=['FP'])
for species in df1_clean['Species'].unique():
    subset = df1_clean[df1_clean['Species'] == species]
    avg_fp = subset.groupby('WTgroup')['FP'].mean()
    plt.plot(avg_fp.index, avg_fp.values, marker='o', label=f'S. {species}', color=colors.get(species, 'black'), linewidth=2)

plt.xticks([3, 4], ['Moist (3)', 'Dry (4)'])
plt.ylabel('Fraction of Evaporation from Rain (FP)')
plt.xlabel('Water Table Condition')
plt.title('Precipitation Dependence Increases in Drought')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_Precipitation_Dependence.png'))
plt.close()

# ---------------------------------------------------------
# Plot 2: VWC vs NEE (Carbon Uptake Tipping Point)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
df3_clean = df3.dropna(subset=['VWC', 'NEE'])
for species in df3_clean['Species'].unique():
    subset = df3_clean[df3_clean['Species'] == species]
    plt.scatter(subset['VWC'], subset['NEE'], label=f'S. {species}', color=colors.get(species, 'black'), alpha=0.7)

plt.axhline(0, color='gray', linestyle='--') # 0 line for Carbon neutrality
plt.xlabel('Volumetric Water Content (VWC)')
plt.ylabel('Net Ecosystem Exchange (NEE)')
plt.title('Carbon Uptake (NEE) vs Moisture (VWC)\nNegative NEE = Absorbing Carbon')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_VWC_vs_NEE.png'))
plt.close()

# ---------------------------------------------------------
# Plot 3: VWC vs Photosystem II Efficiency (PSII)
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
df4_clean = df4.dropna(subset=['VWC', 'PSII'])
for species in df4_clean['Species'].unique():
    subset = df4_clean[df4_clean['Species'] == species]
    plt.scatter(subset['VWC'], subset['PSII'], label=f'S. {species}', color=colors.get(species, 'black'), alpha=0.7)

plt.xlabel('Volumetric Water Content (VWC)')
plt.ylabel('Photosynthetic Efficiency (Fv/Fm)')
plt.title('Photosynthesis Crash Point')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_VWC_vs_PSII.png'))
plt.close()

# ---------------------------------------------------------
# Plot 4: Recovery (Wet vs Rewet)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
df5_clean = df5.dropna(subset=['NEE_Wet', 'NEE_REWET'])

# Calculate averages for bar chart
recovery_means = df5_clean.groupby('Species')[['NEE_Wet', 'NEE_REWET']].mean()
x = np.arange(len(recovery_means))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(x - width/2, recovery_means['NEE_Wet'], width, label='Pre-Drought (Wet)', color='#1E90FF')
ax.bar(x + width/2, recovery_means['NEE_REWET'], width, label='Post-Drought (Rewet)', color='#FFA500')

ax.set_ylabel('Net Ecosystem Exchange (NEE)')
ax.set_title('Carbon Uptake Recovery After Drought')
ax.set_xticks(x)
ax.set_xticklabels([f'S. {s}' for s in recovery_means.index])
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_Drought_Recovery.png'))
plt.close()

print(f"All visualizations saved successfully in the '{output_dir}' folder!")