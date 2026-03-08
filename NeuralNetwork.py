import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ARIMA import test_sip, test_wig, train_sip, train_wig

scaler = StandardScaler()
X_train_sip = scaler.fit_transform(train_sip.drop(columns=["log_return"]))
y_train_sip = train_sip["log_return"]

X_test_sip = scaler.transform(test_sip.drop(columns=["log_return"]))
y_test_sip = test_sip["log_return"]

X_train_wig = scaler.fit_transform(train_wig.drop(columns=["log_return"]))
y_train_wig = train_wig["log_return"]

X_test_wig = scaler.transform(test_wig.drop(columns=["log_return"]))
y_test_wig = test_wig["log_return"]


class ModelDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_data_sip = ModelDataset(X_train_sip, y_train_sip)
test_data_sip = ModelDataset(X_test_sip, y_test_sip)

train_data_wig = ModelDataset(X_train_wig, y_train_wig)
test_data_wig = ModelDataset(X_test_wig, y_test_wig)

train_loader_sip = DataLoader(train_data_sip, batch_size=32, shuffle=False)
test_loader_sip = DataLoader(test_data_sip, batch_size=32, shuffle=False)

train_loader_wig = DataLoader(train_data_wig, batch_size=32, shuffle=False)
test_loader_wig = DataLoader(test_data_wig, batch_size=32, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.activ1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.activ2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 16)
        self.activ3 = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)
        x = self.layer3(x)
        x = self.activ3(x)
        x = self.output_layer(x)
        return x


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


model_sip = NeuralNetwork(X_train_sip.shape[1])
model_wig = NeuralNetwork(X_train_wig.shape[1])

criterion = nn.MSELoss()

optimizer_sip = torch.optim.Adam(model_sip.parameters(), lr=0.001)
optimizer_wig = torch.optim.Adam(model_wig.parameters(), lr=0.001)

train_loss_sip = []
validation_loss_sip = []
for epoch in range(30):
    model_sip.train()
    running_loss_sip = 0.0
    for features, labels in train_loader_sip:
        optimizer_sip.zero_grad()
        outputs = model_sip(features.float()).squeeze()
        loss_sip = criterion(outputs, labels)
        loss_sip.backward()
        optimizer_sip.step()
        running_loss_sip += loss_sip.item()
    running_loss_sip /= len(train_loader_sip.dataset)

    model_sip.eval()
    all_predictions_sip = []
    all_labels_sip = []
    val_loss_sip = 0.0
    with torch.no_grad():
        for features, labels in test_loader_sip:
            outputs = model_sip(features.float()).squeeze()
            loss_sip = criterion(outputs, labels)
            val_loss_sip += loss_sip.item()
            all_predictions_sip.extend(outputs.numpy())
            all_labels_sip.extend(labels.numpy())

    val_loss_sip = val_loss_sip / len(test_loader_sip.dataset)
    train_loss_sip.append(running_loss_sip)
    validation_loss_sip.append(val_loss_sip)
    early_stopping = EarlyStopping(patience=5)
    early_stopping.step(val_loss_sip)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    print(
        f"Epoch [{epoch + 1}/30], Training Loss: {running_loss_sip:.8f}, Validation Loss: {val_loss_sip:.8f}"
    )

# Loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss_sip, label="Train Loss")
plt.plot(validation_loss_sip, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

train_loss_wig = []
validation_loss_wig = []
for epoch in range(30):
    model_sip.train()
    running_loss_wig = 0.0
    for features, labels in train_loader_wig:
        optimizer_wig.zero_grad()
        outputs = model_wig(features.float()).squeeze()
        loss_wig = criterion(outputs, labels)
        loss_wig.backward()
        optimizer_wig.step()
        running_loss_wig += loss_wig.item()
    running_loss_wig /= len(train_loader_wig.dataset)

    model_wig.eval()
    all_predictions_wig = []
    all_labels_wig = []
    val_loss_wig = 0.0
    with torch.no_grad():
        for features, labels in test_loader_wig:
            outputs = model_wig(features.float()).squeeze()
            loss_wig = criterion(outputs, labels.long())
            val_loss_wig += loss_wig.item()
            all_predictions_wig.extend(outputs.numpy())
            all_labels_wig.extend(labels.numpy())

    val_loss_wig = val_loss_wig / len(test_loader_wig.dataset)
    train_loss_wig.append(running_loss_wig)
    validation_loss_wig.append(val_loss_wig)
    early_stopping = EarlyStopping(patience=5)
    early_stopping.step(val_loss_wig)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    print(
        f"Epoch [{epoch + 1}/30], Training Loss: {running_loss_wig:.8f}, Validation Loss: {val_loss_wig:.8f}"
    )

# Loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss_wig, label="Train Loss")
plt.plot(validation_loss_wig, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

mse_sip = mean_squared_error(all_labels_sip, all_predictions_sip)
rmse_sip = root_mean_squared_error(all_labels_sip, all_predictions_sip)
print("S&P MSE:", mse_sip)
print("S&P RMSE:", rmse_sip)

mse_wig = mean_squared_error(all_labels_wig, all_predictions_wig)
rmse_wig = root_mean_squared_error(all_labels_wig, all_predictions_wig)
print("WiG MSE:", mse_wig)
print("WiG RMSE:", rmse_wig)