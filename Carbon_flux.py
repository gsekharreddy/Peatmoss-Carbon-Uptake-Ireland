import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Load & Clean Data
df = pd.read_csv('Dataset 2_Carbon_fluxes.dat', sep='\t')
df = df[df['NEE'] != -999] # Remove missing values

# 2. Feature Engineering
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

# Features: Species, Rain Frequency, Water Table state
X = df[['Species_encoded', 'Freq', 'WTgroup']]
y = df['NEE']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training (XGBoost)
# XGBoost is great here because it handles tabular, non-linear features easily
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
predictions = model.predict(X_test)
print("--- XGBoost NEE Prediction ---")
print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

# --- NEW: Save Model, Metrics, and Confusion Matrix ---
os.makedirs('models', exist_ok=True)

# 1. Save Model & Metrics
joblib.dump(model, 'models/xgboost_nee.joblib')
with open('models/xgboost_metrics.txt', 'w') as f:
    f.write(f"MSE: {mean_squared_error(y_test, predictions):.4f}\nR2: {r2_score(y_test, predictions):.4f}")

# 2. Generate Confusion Matrix (Hack: Classify as Sink vs Source)
# True = Sink (NEE < 0), False = Source (NEE >= 0)
y_test_class = y_test < 0
pred_class = predictions < 0
cm = confusion_matrix(y_test_class, pred_class)

# 3. Plot and Save Graph
plt.figure(figsize=(6, 4))
custom_cmap = LinearSegmentedColormap.from_list('custom', ['#FFA500', '#1E90FF'])
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, 
            xticklabels=['Source', 'Sink'], yticklabels=['Source', 'Sink'])
plt.title('XGBoost: Carbon Sink vs Source Prediction')
plt.ylabel('Actual State')
plt.xlabel('Predicted State')
plt.savefig('models/xgboost_conf_matrix.png')
plt.close()