import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import train_loader, test_loader


class BostonHousingModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BostonHousingModel, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_size)
        self.hl2 = nn.Linear(hidden_size, hidden_size)
        self.op = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = self.op(x)

        return x

model = BostonHousingModel(3, 32)

opt=optim.Adam(model.parameters(), lr=0.001)
criteria=nn.MSELoss()

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        opt.zero_grad()
        outputs = model(X_batch)
        loss = criteria(outputs, y_batch)
        loss.backward()
        opt.step()

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criteria(outputs, y_batch.view(-1, 1))
            test_loss += loss.item() * X_batch.size(0)

    test_loss = test_loss / len(test_loader.dataset)
    rmse = test_loss**0.5

    print(f"Epoch: {epoch+1}, RMSE: {rmse:.3f}")
