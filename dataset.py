import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

data = pd.read_csv('data/Iris.csv')

# Удаление лишних столпцов
data = data.drop(['Id'], axis=1)

# Определние фич
X = data.drop(['Species'], axis=1).values
y = data['Species'].values

# LabelEncoder для "y" фичи
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Создаение класса датасета
class IrisDataset(Dataset):
    def __init__(self, features, target):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.long)
        self.n_samples = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.n_samples

# train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаение объектов датасета
train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

# Создаение лоудеров для батчей
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

for batch_X, batch_y in train_loader:
    print(batch_X.shape)
    print(batch_y.shape)
    break