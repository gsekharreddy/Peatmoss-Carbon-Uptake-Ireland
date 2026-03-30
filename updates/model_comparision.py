import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading Dataset 2 (Carbon Fluxes)...")
df = pd.read_csv('Dataset 2_Carbon_fluxes.dat', sep='\t')
df = df[df['NEE'] != -999]  # Drop missing values

# Encode categorical 'Species'
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

# Features and Target
X = df[['Species_encoded', 'Freq', 'WTgroup']].values
y = df['NEE'].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for SVR and LSTM)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Dictionary to hold our R2 scores
results = {}

# ==========================================
# 2. RANDOM FOREST
# ==========================================
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
results['Random Forest'] = r2_score(y_test, rf_preds)

# ==========================================
# 3. XGBOOST
# ==========================================
print("Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
results['XGBoost'] = r2_score(y_test, xgb_preds)

# ==========================================
# 4. SUPPORT VECTOR REGRESSION (SVR)
# ==========================================
print("Training SVR...")
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
svr_preds = svr_model.predict(X_test_scaled)
results['SVR'] = r2_score(y_test, svr_preds)

# ==========================================
# 5. PYTORCH LSTM (Adapted for Tabular Data)
# ==========================================
print("Training PyTorch LSTM...")

# Scale Target for Neural Network stability
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Reshape X for LSTM: (Batch Size, Sequence Length, Features) -> Seq Len is 1 here
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
    
# Inverse transform to get actual NEE values back
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled).flatten() # Flattened for dataframe formatting
results['PyTorch LSTM'] = r2_score(y_test, lstm_preds)

# ==========================================
# 6. RESULTS OUTPUT & SAVING
# ==========================================
print("\n" + "="*40)
print("🏆 NEE PREDICTION MODEL SHOWDOWN 🏆")
print("="*40)
# Sort results by R2 score (highest to lowest)
sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

for model_name, r2 in sorted_results.items():
    print(f"{model_name:<15} | R2 Score: {r2:.4f}")
print("="*40)

# Compile predictions into a DataFrame for easy viewing
predictions_df = pd.DataFrame({
    'Actual NEE': y_test,
    'XGBoost': xgb_preds,
    'Random Forest': rf_preds,
    'PyTorch LSTM': lstm_preds,
    'SVR': svr_preds
})

# Save the accuracies and sample predictions to a text file
print("\nSaving results to 'model_accuracies.txt'...")
with open('model_accuracies.txt', 'w', encoding='utf-8') as f:
    f.write("🏆 NEE PREDICTION MODEL SHOWDOWN 🏆\n")
    f.write("="*60 + "\n")
    for model_name, r2 in sorted_results.items():
        f.write(f"{model_name:<15} | R2 Score: {r2:.4f}\n")
    f.write("="*60 + "\n\n")
    
    f.write("📊 SAMPLE PREDICTIONS (First 15 Test Cases) 📊\n")
    f.write("="*60 + "\n")
    # Write the first 15 rows of the dataframe to the text file
    f.write(predictions_df.head(15).to_string(index=False))
    f.write("\n" + "="*60 + "\n")
    
print("Done! 📝 Check 'model_accuracies.txt' for the stats and sample predictions.")