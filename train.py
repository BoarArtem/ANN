import torch
from torch import nn

from dataset import loader


class IrisModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(IrisModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))

        return out

model = IrisModel(4, 3)

epochs = 1000

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    model.train()

    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.float()
        y_batch = y_batch.long()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optim.zero_grad()
        loss.backward()
        optim.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = correct / total

    print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")