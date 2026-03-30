import pandas as pd
import numpy as np
import os

# ==========================================
# 1. LOAD AND PREP ORIGINAL DATA
# ==========================================
print("Loading original dataset...")
df = pd.read_csv('Dataset 2_Carbon_fluxes.dat', sep='\t')
df = df[df['NEE'] != -999]

# Pivot to get the Drought Sequence: Wet(2) -> Moist(3) -> Dry(4)
# This matches the exact format needed for Drought_Seq.py
pivot_df = df.pivot_table(index='Core_ID', columns='WTgroup', values='NEE', aggfunc='mean').dropna()

# Extract just the sequence columns (Wet, Moist, Dry)
original_data = pivot_df[[2, 3, 4]].values
num_original = len(original_data)
print(f"Original sequence samples: {num_original}")

# ==========================================
# 2. AMPLIFICATION SETTINGS
# ==========================================
AMPLIFY_FACTOR = 10  # How many times larger the new dataset should be (e.g., 10x)
NOISE_LEVEL = 0.05   # 5% standard deviation noise to mimic natural sensor/field variance

synthetic_data = []

print(f"Amplifying dataset by {AMPLIFY_FACTOR}x using Mixup + Gaussian Noise...")

# Calculate standard deviation for Wet, Moist, and Dry columns
std_devs = np.std(original_data, axis=0)

# ==========================================
# 3. GENERATE SYNTHETIC DATA
# ==========================================
for _ in range(num_original * AMPLIFY_FACTOR):
    # Technique 1: Mixup (Interpolate between two random real samples)
    idx1, idx2 = np.random.choice(num_original, 2, replace=False)
    
    # Use a Beta distribution to decide how much of sample 1 vs sample 2 to use
    alpha = np.random.beta(0.5, 0.5) 
    mixed_sample = (alpha * original_data[idx1]) + ((1 - alpha) * original_data[idx2])
    
    # Technique 2: Gaussian Noise (Add natural jitter)
    # This prevents the network from just memorizing exact interpolations
    noise = np.random.normal(0, std_devs * NOISE_LEVEL)
    
    synthetic_sample = mixed_sample + noise
    synthetic_data.append(synthetic_sample)

synthetic_data = np.array(synthetic_data)

# ==========================================
# 4. PACKAGE AND SAVE
# ==========================================
# Combine original and synthetic arrays
augmented_data = np.vstack((original_data, synthetic_data))

# Convert back to a clean Pandas DataFrame
aug_df = pd.DataFrame(augmented_data, columns=['NEE_Wet', 'NEE_Moist', 'NEE_Dry'])

# Add a flag so you can track which ones are fake vs real during analysis
aug_df['Is_Synthetic'] = [False] * num_original + [True] * len(synthetic_data)

# Save to CSV
output_filename = 'Augmented_Drought_Sequence.csv'
aug_df.to_csv(output_filename, index=False)

print("\n" + "="*40)
print("✅ AMPLIFICATION COMPLETE!")
print("="*40)
print(f"Original size : {num_original} samples")
print(f"New size      : {len(aug_df)} samples")
print(f"Saved to      : {output_filename}")
print("="*40)