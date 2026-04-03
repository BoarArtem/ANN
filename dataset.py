import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/Iris.csv")
data = data.drop(['Id'], axis=1)

X = data.drop(['Species'], axis=1).values
y = data['Species']
y = y.map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})

scaler = StandardScaler()
X = scaler.fit_transform(X)

class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).long()
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

dataset = IrisDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    print(X_batch.shape)
    print(y_batch.shape)

    break
