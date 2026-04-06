"""
Model Prediction Correlation Matrix
===================================
Trains the 4 regression models used in the advanced evaluation folder and
creates a correlation matrix from their test-set predictions.

Outputs:
- plots/model_prediction_correlation_matrix.csv
- plots/model_prediction_correlation_matrix.png
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="darkgrid", context="talk")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset 2_Carbon_fluxes.dat")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")


class TabularLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df[df["NEE"] != -999].copy()

    encoder = LabelEncoder()
    df["Species_encoded"] = encoder.fit_transform(df["Species"])

    features = df[["Species_encoded", "Freq", "WTgroup"]].values
    target = df["NEE"].values
    return features, target


def train_models(X_train_scaled, X_test_scaled, y_train):
    predictions = {}

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    predictions["Random Forest"] = rf_model.predict(X_test_scaled)

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train_scaled, y_train)
    predictions["XGBoost"] = xgb_model.predict(X_test_scaled)

    svr_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    svr_model.fit(X_train_scaled, y_train)
    predictions["SVR"] = svr_model.predict(X_test_scaled)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    lstm_model = TabularLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)

    lstm_model.train()
    for _ in range(150):
        optimizer.zero_grad()
        outputs = lstm_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_preds_scaled = lstm_model(X_test_tensor).cpu().numpy()

    lstm_predictions = scaler_y.inverse_transform(lstm_preds_scaled).flatten()
    predictions["PyTorch LSTM"] = lstm_predictions

    return predictions


def plot_correlation_matrix(correlation_matrix):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Model Predictions", fontweight="bold")
    plt.tight_layout()
    return fig


def main():
    print("Loading Dataset 2 (Carbon Fluxes)...")
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    print("Training models...")
    predictions = train_models(X_train_scaled, X_test_scaled, y_train)

    prediction_frame = pd.DataFrame(predictions)
    correlation_matrix = prediction_frame.corr(method="pearson")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "model_prediction_correlation_matrix.csv")
    correlation_matrix.to_csv(csv_path)
    print(f"Saved correlation matrix CSV: {csv_path}")

    fig = plot_correlation_matrix(correlation_matrix)
    image_path = os.path.join(OUTPUT_DIR, "model_prediction_correlation_matrix.png")
    fig.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correlation matrix heatmap: {image_path}")

    print("\nCorrelation matrix:")
    print(correlation_matrix.to_string())

    print("\nDone. The matrix compares the four models using their test-set predictions.")


if __name__ == "__main__":
    main()