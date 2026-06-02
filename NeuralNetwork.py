import os
import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from preprocessing import df_sip, df_wig

# create train/test splits (daty zgodnie z pozostałymi skryptami)
train_sip = df_sip.loc[df_sip.index < "2025-01-01"]
test_sip = df_sip.loc[df_sip.index >= "2025-01-01"]
train_wig = df_wig.loc[df_wig.index < "2025-01-01"]
test_wig = df_wig.loc[df_wig.index >= "2025-01-01"]

# Create separate scalers for SIP and WIG (do not overwrite each other)
scaler_sip = StandardScaler()
X_train_sip = scaler_sip.fit_transform(train_sip.drop(columns=["log_return"]))
y_train_sip = train_sip["log_return"]

X_test_sip = scaler_sip.transform(test_sip.drop(columns=["log_return"]))
y_test_sip = test_sip["log_return"]

scaler_wig = StandardScaler()
X_train_wig = scaler_wig.fit_transform(train_wig.drop(columns=["log_return"]))
y_train_wig = train_wig["log_return"]

X_test_wig = scaler_wig.transform(test_wig.drop(columns=["log_return"]))
y_test_wig = test_wig["log_return"]

# Sequence length for LSTM (sliding window). Możesz dostosować do eksperymentów.
SEQ_LEN = 10


class ModelDataset(Dataset):
    """Dataset zwracający sekwencje (sliding window) i odpowiadające cele (następny krok).
    Wejście X: numpy array o kształcie (N, num_features)
    y: pandas Series długości N
    Długość datasetu = N - SEQ_LEN
    """

    def __init__(self, X, y, seq_len=SEQ_LEN):
        super().__init__()
        import numpy as _np

        self.seq_len = seq_len
        # Ensure numpy array
        if hasattr(X, "values"):
            X = X.values
        self.X = _np.asarray(X, dtype=_np.float32)
        # y may be pandas Series -> convert to numpy
        if hasattr(y, "values"):
            y = y.values
        self.y = _np.asarray(y, dtype=_np.float32)
        self.num_samples = max(0, len(self.X) - self.seq_len)

        if len(self.X) <= self.seq_len:
            raise ValueError(
                f"Input sequence length ({len(self.X)}) must be greater than seq_len ({self.seq_len}). "
                "Increase your data length or reduce SEQ_LEN."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # return sequence of shape (sliding window, num_features) and target scalar
        start = idx
        end = idx + self.seq_len
        seq = self.X[start:end]
        target = self.y[end]  # predict next step after sequence
        # Convert to torch tensors
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# Create dataset objects (kept at module level so XGBoost can import the arrays)
train_data_sip = ModelDataset(X_train_sip, y_train_sip)
# wrap test dataset creation in try/except in case test set too small
try:
    test_data_sip = ModelDataset(X_test_sip, y_test_sip)
except ValueError:
    test_data_sip = None

train_data_wig = ModelDataset(X_train_wig, y_train_wig)
try:
    test_data_wig = ModelDataset(X_test_wig, y_test_wig)
except ValueError:
    test_data_wig = None

train_loader_sip = DataLoader(train_data_sip, batch_size=32, shuffle=False)
if test_data_sip is not None:
    test_loader_sip = DataLoader(test_data_sip, batch_size=32, shuffle=False)
else:
    test_loader_sip = None

train_loader_wig = DataLoader(train_data_wig, batch_size=32, shuffle=False)
if test_data_wig is not None:
    test_loader_wig = DataLoader(test_data_wig, batch_size=32, shuffle=False)
else:
    test_loader_wig = None


class NeuralNetwork(nn.Module):
    """LSTM-based network. Zachowuje nazwę "NeuralNetwork" aby nie zmieniać pozostałego kodu.
    Przyjmuje wejście kształtu (batch, seq_len, input_size) i zwraca (batch, 1).
    """

    def __init__(self, input_size, hidden_size=64, num_layers=1, seq_len=SEQ_LEN):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Final fully connected layer to scalar output
        self.fc = nn.Linear(hidden_size, 1)

        # Optional: attach scalers placeholders to model instance if needed later
        self.scaler = None

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # LSTM returns output (batch, seq_len, hidden_size) and (h_n, c_n)
        out, (h_n, c_n) = self.lstm(x)
        # use last time-step output
        last = out[:, -1, :]
        out = self.fc(last)
        return out.squeeze(-1)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# Training function with checkpointing
def run_neural_network(epochs=30, lr=0.001, batch_size=32, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    model_sip = NeuralNetwork(X_train_sip.shape[1])
    model_wig = NeuralNetwork(X_train_wig.shape[1])

    # attach appropriate scalers to models for clarity
    model_sip.scaler = scaler_sip
    model_wig.scaler = scaler_wig

    criterion = nn.MSELoss()

    optimizer_sip = torch.optim.Adam(model_sip.parameters(), lr=lr)
    optimizer_wig = torch.optim.Adam(model_wig.parameters(), lr=lr)

    # SIP training
    train_loss_sip = []
    validation_loss_sip = []
    early_stopping_sip = EarlyStopping(patience=5)
    best_val_sip = float("inf")
    for epoch in range(epochs):
        model_sip.train()
        running_loss_sip = 0.0
        for features, labels in train_loader_sip:
            optimizer_sip.zero_grad()
            outputs = model_sip(features.float()).squeeze()
            loss_sip = criterion(outputs, labels)
            loss_sip.backward()
            optimizer_sip.step()
            running_loss_sip += loss_sip.item()
        running_loss_sip /= len(train_loader_sip)

        model_sip.eval()
        all_predictions_sip = []
        all_labels_sip = []
        val_loss_sip = 0.0
        if test_loader_sip is None:
            print("No test/validation data for SIP; skipping validation.")
            val_loss_sip = float("inf")
        else:
            with torch.no_grad():
                for features, labels in test_loader_sip:
                    outputs = model_sip(features.float()).squeeze()
                    loss_sip = criterion(outputs, labels)
                    val_loss_sip += loss_sip.item()
                    all_predictions_sip.extend(outputs.numpy())
                    all_labels_sip.extend(labels.numpy())
            val_loss_sip = val_loss_sip / len(test_loader_sip)

        train_loss_sip.append(running_loss_sip)
        validation_loss_sip.append(val_loss_sip)

        # checkpoint
        if val_loss_sip < best_val_sip:
            best_val_sip = val_loss_sip
            torch.save(model_sip.state_dict(), os.path.join(save_dir, "best_model_sip.pth"))
            # save scaler
            with open(os.path.join(save_dir, "scaler_sip.pkl"), "wb") as f:
                pickle.dump(scaler_sip, f)

        early_stopping_sip.step(val_loss_sip)
        if early_stopping_sip.should_stop:
            print(f"Early stopping SIP at epoch {epoch + 1}")
            break

        print(
            f"SIP Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss_sip:.8f}, Validation Loss: {val_loss_sip:.8f}"
        )

    # plot losses for SIP
    if len(train_loss_sip) > 0 and validation_loss_sip and validation_loss_sip[0] != float("inf"):
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_sip, label="Train Loss")
        plt.plot(validation_loss_sip, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SIP Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    # WIG training
    train_loss_wig = []
    validation_loss_wig = []
    early_stopping_wig = EarlyStopping(patience=5)
    best_val_wig = float("inf")
    for epoch in range(epochs):
        model_wig.train()
        running_loss_wig = 0.0
        for features, labels in train_loader_wig:
            optimizer_wig.zero_grad()
            outputs = model_wig(features.float()).squeeze()
            loss_wig = criterion(outputs, labels)
            loss_wig.backward()
            optimizer_wig.step()
            running_loss_wig += loss_wig.item()
        running_loss_wig /= len(train_loader_wig)

        model_wig.eval()
        all_predictions_wig = []
        all_labels_wig = []
        val_loss_wig = 0.0
        if test_loader_wig is None:
            print("No test/validation data for WIG; skipping validation.")
            val_loss_wig = float("inf")
        else:
            with torch.no_grad():
                for features, labels in test_loader_wig:
                    outputs = model_wig(features.float()).squeeze()
                    loss_wig = criterion(outputs, labels)
                    val_loss_wig += loss_wig.item()
                    all_predictions_wig.extend(outputs.numpy())
                    all_labels_wig.extend(labels.numpy())
            val_loss_wig = val_loss_wig / len(test_loader_wig)

        train_loss_wig.append(running_loss_wig)
        validation_loss_wig.append(val_loss_wig)

        # checkpoint
        if val_loss_wig < best_val_wig:
            best_val_wig = val_loss_wig
            torch.save(model_wig.state_dict(), os.path.join(save_dir, "best_model_wig.pth"))
            with open(os.path.join(save_dir, "scaler_wig.pkl"), "wb") as f:
                pickle.dump(scaler_wig, f)

        early_stopping_wig.step(val_loss_wig)
        if early_stopping_wig.should_stop:
            print(f"Early stopping WIG at epoch {epoch + 1}")
            break

        print(
            f"WIG Epoch [{epoch + 1}/{epochs}], Training Loss: {running_loss_wig:.8f}, Validation Loss: {val_loss_wig:.8f}"
        )

    # plot losses for WIG
    if len(train_loss_wig) > 0 and validation_loss_wig and validation_loss_wig[0] != float("inf"):
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_wig, label="Train Loss")
        plt.plot(validation_loss_wig, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("WIG Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    # final metrics (if predictions exist)
    results = {}
    if 'all_predictions_sip' in locals() and len(all_predictions_sip) > 0:
        mse_sip = mean_squared_error(all_labels_sip, all_predictions_sip)
        rmse_sip = (mse_sip ** 0.5)
        results['sip'] = {'mse': mse_sip, 'rmse': rmse_sip}
        print("S&P MSE NN:", mse_sip)
        print("S&P RMSE NN:", rmse_sip)

    if 'all_predictions_wig' in locals() and len(all_predictions_wig) > 0:
        mse_wig = mean_squared_error(all_labels_wig, all_predictions_wig)
        rmse_wig = (mse_wig ** 0.5)
        results['wig'] = {'mse': mse_wig, 'rmse': rmse_wig}
        print("WiG MSE NN:", mse_wig)
        print("WiG RMSE NN:", rmse_wig)

    return results


if __name__ == "__main__":
    run_neural_network()
