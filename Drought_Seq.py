import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# 1. Load & Engineer Sequential Data
df = pd.read_csv('Dataset 2_Carbon_fluxes.dat', sep='\t')
df = df[df['NEE'] != -999]

# Create a time-series sequence: Wet(2) -> Moist(3) -> Dry(4)
# We will use NEE at Wet and Moist to predict NEE at Dry for each core
# FIX: Used pivot_table instead of pivot to handle duplicate entries by averaging them
pivot_df = df.pivot_table(index='Core_ID', columns='WTgroup', values='NEE', aggfunc='mean').dropna()

# Input Sequence: [Wet (2), Moist (3)] -> Target: [Dry (4)]
X_seq = pivot_df[[2, 3]].values
y_target = pivot_df[4].values

# 2. Scale & Reshape for PyTorch LSTM: (Batch Size, Sequence Length, Features)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_seq).reshape(-1, 2, 1) # Seq len = 2, Features = 1

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1))

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 3. Define the PyTorch LSTM Model
class PeatmossLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super(PeatmossLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out shape: (batch_size, seq_length, hidden_size)
        out, (hn, cn) = self.lstm(x)
        # We only want the output from the last time step
        last_time_step_out = out[:, -1, :]
        prediction = self.fc(last_time_step_out)
        return prediction

# 4. Initialize Model, Loss, and Optimizer
model = PeatmossLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Training Loop
epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. Evaluation
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)
    
    # Calculate R2 (working with numpy arrays)
    y_test_np = y_test.numpy()
    test_preds_np = test_preds.numpy()
    ss_res = np.sum((y_test_np - test_preds_np) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

print("\n--- PyTorch LSTM Drought Prediction ---")
print(f"Test MSE (Scaled): {test_loss.item():.4f}")
print(f"Test R2 Score: {r2:.4f}")

# --- NEW: Save Model, Metrics, and Confusion Matrix ---
os.makedirs('models', exist_ok=True)

# 1. Save Model & Metrics
torch.save(model.state_dict(), 'models/lstm_drought.pth')
with open('models/lstm_metrics.txt', 'w') as f:
    f.write(f"MSE: {test_loss.item():.4f}\nR2: {r2:.4f}")

# 2. Generate Confusion Matrix (Hack: Classify as Sink vs Source)
# We have to reverse the scaling to check if NEE < 0 (Carbon Sink)
y_test_unscaled = scaler_y.inverse_transform(y_test_np)
preds_unscaled = scaler_y.inverse_transform(test_preds_np)

y_test_class = y_test_unscaled < 0
pred_class = preds_unscaled < 0
cm = confusion_matrix(y_test_class, pred_class)

# 3. Plot and Save Graph
plt.figure(figsize=(6, 4))
custom_cmap = LinearSegmentedColormap.from_list('custom', ['#FFA500', '#1E90FF'])
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, 
            xticklabels=['Source', 'Sink'], yticklabels=['Source', 'Sink'])
plt.title('LSTM: Drought End-State Prediction')
plt.ylabel('Actual State')
plt.xlabel('Predicted State')
plt.savefig('models/lstm_conf_matrix.png')
plt.close()