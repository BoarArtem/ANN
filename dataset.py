import pandas as pd
import torch
from torch import nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/bitcoin.csv")
data = data.drop(['Date'], axis=1)

X = data.iloc[:, 0:-1].values

y = data.iloc[:, -1]
y = y.map({"Low": 0, "Medium": 1, "High": 2})

scaler = StandardScaler()
X = scaler.fit_transform(X)

# X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

class BitcoinDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).long()
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)

dataset = BitcoinDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32)

for X, y in loader:
    print(X.shape), print(y.shape)

    break