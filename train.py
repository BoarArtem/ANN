import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import train_loader, test_loader


class BitcoinModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BitcoinModel, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_size)
        self.hl2 = nn.Linear(hidden_size, hidden_size)
        self.op = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = self.op(x)

        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, path="bitcoin_best.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

model = BitcoinModel(5, 64, 3)
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

optimization = optim.Adam(model.parameters(), lr=0.01)
criteria = nn.CrossEntropyLoss()

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimization.zero_grad()
        outputs = model(X_batch)
        loss = criteria(outputs, y_batch)
        loss.backward()
        optimization.step()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)

            loss = criteria(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_loss = val_loss / len(test_loader.dataset)
    accuracy = correct / total

    print(f"Epoch: {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.3f}")

    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

model.load_state_dict(torch.load("bitcoin_best.pth"))