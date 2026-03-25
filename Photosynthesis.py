import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

# 1. Load & Clean Data (This is the part that was missing!)
df = pd.read_csv('Dataset 4_VWCvsPSII.dat', sep='\t')
df = df[df['PSII'] != -999]

# 2. Feature Engineering
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

X = df[['Species_encoded', 'VWC']]
y = df['PSII']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling 
# (CRITICAL for SVR: Support Vector Machines are highly sensitive to unscaled data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training (Support Vector Regression)
# RBF kernel handles the non-linear "tipping point" of photosynthesis dropping off
model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
model.fit(X_train_scaled, y_train)

# 6. Evaluation
predictions = model.predict(X_test_scaled)
print("--- SVR PSII Prediction ---")
print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

# --- Save Model, Metrics, and Confusion Matrix ---
os.makedirs('models', exist_ok=True)

# Save Model & Metrics
joblib.dump(model, 'models/svr_psii.joblib')
with open('models/svr_metrics.txt', 'w') as f:
    f.write(f"MSE: {mean_squared_error(y_test, predictions):.4f}\nR2: {r2_score(y_test, predictions):.4f}")

# Generate Confusion Matrix (Hack: Active vs Inactive Photosynthesis)
# Based on the paper, PSII crashes significantly around 0.49 VWC
y_test_class = y_test > 0.49
pred_class = predictions > 0.49
cm = confusion_matrix(y_test_class, pred_class)

# Plot and Save Graph
plt.figure(figsize=(6, 4))
custom_cmap = LinearSegmentedColormap.from_list('custom', ['#FFA500', '#1E90FF'])
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, 
            xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
plt.title('SVR: Photosynthesis State Prediction')
plt.ylabel('Actual State')
plt.xlabel('Predicted State')
plt.savefig('models/svr_conf_matrix.png')
plt.close()