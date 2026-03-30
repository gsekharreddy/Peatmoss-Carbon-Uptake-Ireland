"""
Extensive Model Evaluation Script
==================================
Comprehensive evaluation of 4 models with multiple metrics including:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- Adjusted R² Score
- Mean Absolute Percentage Error (MAPE)
- Median Absolute Error
- Cross-validation scores
- Performance ranking

Models evaluated: XGBoost, Random Forest, SVR, PyTorch LSTM
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error
)
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("="*70)
print("EXTENSIVE MODEL EVALUATION FRAMEWORK")
print("="*70)
print("\n📊 Loading and preprocessing data...")

df = pd.read_csv('../../Dataset 2_Carbon_fluxes.dat', sep='\t')
df = df[df['NEE'] != -999]  # Drop missing values

# Encode categorical 'Species'
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

# Features and Target
X = df[['Species_encoded', 'Freq', 'WTgroup']].values
y = df['NEE'].values

print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Target variable shape: {y.shape}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale Target for Neural Network stability
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# ==========================================
# METRICS CALCULATION FUNCTIONS
# ==========================================

class ModelEvaluator:
    """Comprehensive model evaluation toolkit"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, model_name="Model", n_features=3):
        """
        Calculate comprehensive metrics for model evaluation
        """
        # Ensure data types are float
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        n_samples = len(y_true)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Adjusted R² Score
        adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        
        # Mean Absolute Percentage Error
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, a_min=1e-10, a_max=None))) * 100
        
        # Median Absolute Error
        medae = median_absolute_error(y_true, y_pred)
        
        # Root Mean Squared Log Error
        rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
        
        # Mean Directional Accuracy (for trends)
        actual_direction = np.sign(np.diff(y_true)).astype(np.float64)
        pred_direction = np.sign(np.diff(y_pred)).astype(np.float64)
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Explained Variance Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_var = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Residuals statistics
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_skewness = 0
        if residual_std != 0:
            residual_skewness = np.mean((residuals - np.mean(residuals)) ** 3) / (residual_std ** 3)
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MSE': mse,
            'R² Score': r2,
            'Adjusted R²': adj_r2,
            'MAPE (%)': mape,
            'Median Abs Error': medae,
            'RMSLE': rmsle,
            'Explained Variance': explained_var,
            'Directional Accuracy (%)': directional_accuracy,
            'Residual Std Dev': residual_std,
            'Residual Skewness': residual_skewness
        }
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X, y, cv_folds=5):
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            if isinstance(model, SVR):
                model.fit(X_train_cv, y_train_cv)
                y_pred_cv = model.predict(X_val_cv)
            elif isinstance(model, (RandomForestRegressor, XGBRegressor)):
                model.fit(X_train_cv, y_train_cv)
                y_pred_cv = model.predict(X_val_cv)
            else:
                # For LSTM
                continue
            
            rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))
            r2_scores.append(r2_score(y_val_cv, y_pred_cv))
            mae_scores.append(mean_absolute_error(y_val_cv, y_pred_cv))
        
        return {
            'CV RMSE Mean': np.mean(rmse_scores),
            'CV RMSE Std': np.std(rmse_scores),
            'CV MAE Mean': np.mean(mae_scores),
            'CV MAE Std': np.std(mae_scores),
            'CV R² Mean': np.mean(r2_scores),
            'CV R² Std': np.std(r2_scores)
        }

# ==========================================
# 2. MODEL TRAINING
# ==========================================
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

models = {}
predictions = {}
all_metrics = []

# Random Forest
print("\n🌲 Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_preds
print("✓ Random Forest trained")

# XGBoost
print("🚀 Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_preds
print("✓ XGBoost trained")

# SVR
print("📊 Training Support Vector Regression...")
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_preds = svr_model.predict(X_test_scaled)
models['SVR'] = svr_model
predictions['SVR'] = svr_preds
print("✓ SVR trained")

# PyTorch LSTM
print("🧠 Training PyTorch LSTM...")
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

lstm_model.train()
for epoch in range(150):
    optimizer.zero_grad()
    outputs = lstm_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

lstm_model.eval()
with torch.no_grad():
    lstm_preds_scaled = lstm_model(X_test_tensor).numpy()

lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten()
models['PyTorch LSTM'] = lstm_model
predictions['PyTorch LSTM'] = lstm_preds
print("✓ PyTorch LSTM trained")

# ==========================================
# 3. COMPREHENSIVE EVALUATION
# ==========================================
print("\n" + "="*70)
print("COMPREHENSIVE EVALUATION")
print("="*70)

for model_name in ['Random Forest', 'XGBoost', 'SVR', 'PyTorch LSTM']:
    print(f"\n📈 Evaluating {model_name}...")
    y_pred = predictions[model_name]
    
    # Calculate basic metrics
    metrics = ModelEvaluator.calculate_all_metrics(y_test, y_pred, model_name)
    all_metrics.append(metrics)
    
    # Cross-validation (skip LSTM as it requires special handling)
    if model_name != 'PyTorch LSTM':
        print(f"   Performing 5-fold cross-validation...")
        cv_metrics = ModelEvaluator.cross_validate_model(models[model_name], X_train_scaled, y_train, cv_folds=5)
        metrics.update(cv_metrics)
    
    print(f"   ✓ {model_name} evaluation complete")

# ==========================================
# 4. RESULTS COMPILATION & REPORTING
# ==========================================
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

results_df = pd.DataFrame(all_metrics)
results_df = results_df.set_index('Model')

# Create organized output
print("\n📊 PRIMARY METRICS (RMSE & MAE)")
print("-" * 70)
primary_metrics = results_df[['RMSE', 'MAE', 'R² Score']]
print(primary_metrics.to_string())

print("\n📊 EXTENDED METRICS")
print("-" * 70)
extended_metrics = results_df[['MSE', 'Adjusted R²', 'MAPE (%)', 'Median Abs Error']]
print(extended_metrics.to_string())

print("\n📊 ADDITIONAL METRICS")
print("-" * 70)
additional_metrics = results_df[['RMSLE', 'Explained Variance', 'Directional Accuracy (%)',
                                   'Residual Std Dev', 'Residual Skewness']]
print(additional_metrics.to_string())

# Cross-validation metrics if available
cv_cols = [col for col in results_df.columns if col.startswith('CV')]
if cv_cols:
    print("\n📊 CROSS-VALIDATION METRICS (5-Fold)")
    print("-" * 70)
    cv_metrics = results_df[cv_cols]
    print(cv_metrics.to_string())

# ==========================================
# 5. RANKING & INSIGHTS
# ==========================================
print("\n" + "="*70)
print("MODEL RANKING & INSIGHTS")
print("="*70)

# Rank by R² Score
ranked_r2 = results_df['R² Score'].sort_values(ascending=False)
print("\n🏆 Models Ranked by R² Score:")
for rank, (model, score) in enumerate(ranked_r2.items(), 1):
    print(f"  {rank}. {model}: {score:.4f}")

# Rank by RMSE (lower is better)
ranked_rmse = results_df['RMSE'].sort_values(ascending=True)
print("\n🎯 Models Ranked by RMSE (Lower is Better):")
for rank, (model, score) in enumerate(ranked_rmse.items(), 1):
    print(f"  {rank}. {model}: {score:.4f}")

# Rank by MAE (lower is better)
ranked_mae = results_df['MAE'].sort_values(ascending=True)
print("\n🎯 Models Ranked by MAE (Lower is Better):")
for rank, (model, score) in enumerate(ranked_mae.items(), 1):
    print(f"  {rank}. {model}: {score:.4f}")

# Best overall model (weighted ranking)
print("\n⭐ OVERALL PERFORMANCE")
print("-" * 70)
# Normalize metrics for scoring
r2_norm = (results_df['R² Score'] - results_df['R² Score'].min()) / (results_df['R² Score'].max() - results_df['R² Score'].min())
rmse_norm = 1 - (results_df['RMSE'] - results_df['RMSE'].min()) / (results_df['RMSE'].max() - results_df['RMSE'].min())
mae_norm = 1 - (results_df['MAE'] - results_df['MAE'].min()) / (results_df['MAE'].max() - results_df['MAE'].min())

overall_score = (r2_norm * 0.5 + rmse_norm * 0.25 + mae_norm * 0.25)
best_model = overall_score.idxmax()
print(f"Best Overall Model (by weighted metrics): {best_model}")
print(f"Overall Score: {overall_score[best_model]:.4f}")

# ==========================================
# 6. SAVE RESULTS
# ==========================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create output directory if it doesn't exist
os.makedirs('reports', exist_ok=True)

# Save detailed results
results_df.to_csv('reports/detailed_evaluation_metrics.csv')
print("✓ Saved: reports/detailed_evaluation_metrics.csv")

# Save sorted rankings
rankings = pd.DataFrame({
    'Rank by R²': ranked_r2.index,
    'R² Score': ranked_r2.values,
    'Rank by RMSE': ranked_rmse.index,
    'RMSE': ranked_rmse.values,
    'Rank by MAE': ranked_mae.index,
    'MAE': ranked_mae.values
}).reset_index(drop=True)

rankings.to_csv('reports/model_rankings.csv', index=False)
print("✓ Saved: reports/model_rankings.csv")

# Generate detailed report
report_path = 'reports/EVALUATION_REPORT.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("EXTENSIVE MODEL EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Total samples: {len(X)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Number of features: {X.shape[1]}\n")
    f.write(f"Target variable: NEE (Net Ecosystem Exchange)\n\n")
    
    f.write("MODELS EVALUATED\n")
    f.write("-"*70 + "\n")
    f.write("1. Random Forest (n_estimators=100)\n")
    f.write("2. XGBoost (n_estimators=200, learning_rate=0.05)\n")
    f.write("3. Support Vector Regression (SVR, kernel=RBF)\n")
    f.write("4. PyTorch LSTM (hidden_size=16, epochs=150)\n\n")
    
    f.write("PRIMARY METRICS (RMSE & MAE)\n")
    f.write("-"*70 + "\n")
    f.write(primary_metrics.to_string())
    f.write("\n\n")
    
    f.write("EXTENDED METRICS\n")
    f.write("-"*70 + "\n")
    f.write(extended_metrics.to_string())
    f.write("\n\n")
    
    f.write("ADDITIONAL METRICS\n")
    f.write("-"*70 + "\n")
    f.write(additional_metrics.to_string())
    f.write("\n\n")
    
    if cv_cols:
        f.write("CROSS-VALIDATION METRICS (5-Fold)\n")
        f.write("-"*70 + "\n")
        f.write(cv_metrics.to_string())
        f.write("\n\n")
    
    f.write("MODEL RANKINGS\n")
    f.write("-"*70 + "\n")
    f.write("\nBy R² Score (Higher is Better):\n")
    for rank, (model, score) in enumerate(ranked_r2.items(), 1):
        f.write(f"  {rank}. {model}: {score:.4f}\n")
    
    f.write("\nBy RMSE (Lower is Better):\n")
    for rank, (model, score) in enumerate(ranked_rmse.items(), 1):
        f.write(f"  {rank}. {model}: {score:.4f}\n")
    
    f.write("\nBy MAE (Lower is Better):\n")
    for rank, (model, score) in enumerate(ranked_mae.items(), 1):
        f.write(f"  {rank}. {model}: {score:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write(f"BEST OVERALL MODEL: {best_model}\n")
    f.write(f"Overall Score: {overall_score[best_model]:.4f}\n")
    f.write("="*70 + "\n")

print(f"✓ Saved: {report_path}")

print("\n" + "="*70)
print("✨ EXTENSIVE EVALUATION COMPLETE ✨")
print("="*70)
print("\nGenerated files in 'reports/' directory:")
print("  - detailed_evaluation_metrics.csv")
print("  - model_rankings.csv")
print("  - EVALUATION_REPORT.txt")
print("\nKey insight: Best model is", best_model)
print("="*70)
