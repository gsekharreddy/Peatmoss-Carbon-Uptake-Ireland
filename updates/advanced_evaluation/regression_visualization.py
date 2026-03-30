"""
Regression Evaluation Visualization Script
============================================
Generates 4 comprehensive regression evaluation plots for each of the 4 models.
Each plot contains:
- Predicted vs Actual scatter plot with R2 score
- Residuals distribution plot
- Q-Q plot for residuals
- Feature importance (where applicable)

Models evaluated: XGBoost, Random Forest, SVR, PyTorch LSTM
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading Dataset 2 (Carbon Fluxes)...")
df = pd.read_csv('../../Dataset 2_Carbon_fluxes.dat', sep='\t')
df = df[df['NEE'] != -999]  # Drop missing values

# Encode categorical 'Species'
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

# Features and Target
X = df[['Species_encoded', 'Freq', 'WTgroup']].values
y = df['NEE'].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale Target for Neural Network stability
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Dictionary to store predictions and models
models = {}
predictions = {}

print("\n" + "="*60)
print("TRAINING MODELS FOR EVALUATION")
print("="*60)

# ==========================================
# 2. RANDOM FOREST
# ==========================================
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_preds

# ==========================================
# 3. XGBOOST
# ==========================================
print("Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_preds

# ==========================================
# 4. SUPPORT VECTOR REGRESSION (SVR)
# ==========================================
print("Training SVR...")
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_preds = svr_model.predict(X_test_scaled)
models['SVR'] = svr_model
predictions['SVR'] = svr_preds

# ==========================================
# 5. PYTORCH LSTM
# ==========================================
print("Training PyTorch LSTM...")

# Reshape X for LSTM: (Batch Size, Sequence Length, Features)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

class TabularLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=16):
        super(TabularLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_model = TabularLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)

# Training loop
lstm_model.train()
for epoch in range(150):
    optimizer.zero_grad()
    outputs = lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluation
lstm_model.eval()
with torch.no_grad():
    lstm_preds_scaled = lstm_model(X_test_tensor).numpy()

# Inverse transform to get actual NEE values
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten()
models['PyTorch LSTM'] = lstm_model
predictions['PyTorch LSTM'] = lstm_preds

print("\n" + "="*60)
print("GENERATING VISUALIZATION PLOTS")
print("="*60)

# Create output directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def plot_regression_evaluation(y_true, y_pred, model_name):
    """
    Create a 2x2 grid plot with:
    - Predicted vs Actual
    - Residuals Distribution
    - Q-Q Plot
    - Residuals vs Predicted
    """
    rmse, mae, r2 = calculate_metrics(y_true, y_pred)
    residuals = y_true - y_pred
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{model_name} - Regression Evaluation Matrix\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual NEE', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted NEE', fontsize=11, fontweight='bold')
    ax1.set_title('Predicted vs Actual Values', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Residuals', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q Plot
    ax3 = fig.add_subplot(gs[1, 0])
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals vs Predicted
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted NEE', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax4.set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    return fig, (rmse, mae, r2)

# ==========================================
# GENERATE PLOTS FOR EACH MODEL
# ==========================================

all_metrics = {}

for model_name in ['XGBoost', 'Random Forest', 'SVR', 'PyTorch LSTM']:
    print(f"\nGenerating plot for {model_name}...")
    y_pred = predictions[model_name]
    
    fig, metrics = plot_regression_evaluation(y_test, y_pred, model_name)
    
    # Save plot
    filename = f"plots/{model_name.lower().replace(' ', '_')}_evaluation.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    all_metrics[model_name] = metrics
    plt.close(fig)

# ==========================================
# SUMMARY TABLE
# ==========================================
print("\n" + "="*60)
print("SUMMARY OF METRICS")
print("="*60)

summary_df = pd.DataFrame(all_metrics).T
summary_df.columns = ['RMSE', 'MAE', 'R²']
summary_df = summary_df.sort_values('R²', ascending=False)

print(summary_df.to_string())

# Save summary to CSV
summary_df.to_csv('plots/metrics_summary.csv')
print("\n✓ Metrics summary saved to: plots/metrics_summary.csv")

# Create a combined comparison figure
print("\nGenerating combined comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE and MAE comparison
ax = axes[0]
x_pos = np.arange(len(summary_df))
width = 0.35

ax.bar(x_pos - width/2, summary_df['RMSE'], width, label='RMSE', alpha=0.8)
ax.bar(x_pos + width/2, summary_df['MAE'], width, label='MAE', alpha=0.8)

ax.set_xlabel('Model', fontsize=11, fontweight='bold')
ax.set_ylabel('Error', fontsize=11, fontweight='bold')
ax.set_title('RMSE vs MAE Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(summary_df.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# R² comparison
ax = axes[1]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(summary_df)))
bars = ax.barh(summary_df.index, summary_df['R²'], color=colors, alpha=0.8, edgecolor='black')

ax.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('Model Performance (R² Score)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (idx, row) in enumerate(summary_df.iterrows()):
    ax.text(row['R²'] + 0.02, i, f"{row['R²']:.4f}", va='center', fontweight='bold')

plt.tight_layout()
fig.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/model_comparison.png")
plt.close(fig)

print("\n" + "="*60)
print("✨ VISUALIZATION COMPLETE ✨")
print("="*60)
print(f"Generated {len(predictions)} model evaluation plots")
print(f"All plots saved in: plots/")
print("Files created:")
print("  - xgboost_evaluation.png")
print("  - random_forest_evaluation.png")
print("  - svr_evaluation.png")
print("  - pytorch_lstm_evaluation.png")
print("  - model_comparison.png")
print("  - metrics_summary.csv")
