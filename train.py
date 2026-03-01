import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from dataset import train_loader, test_loader

# Создаение класса модели
class IrisModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisModel, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_size) # hidden layer 1
        self.hl2 = nn.Linear(hidden_size, hidden_size) # hidden layer 2
        self.ol = nn.Linear(hidden_size, num_classes) # output layer

    def forward(self, x):
        x = F.relu(self.hl1(x)) # relu for hl1
        x = F.relu(self.hl2(x)) # relu for hl1
        x = self.ol(x) # nothing one for o/p layer

        return x

# Создание объекта модели
model = IrisModel(input_size=4, hidden_size=64, num_classes=3)

# optimizer and lasso
opt=Adam(model.parameters(), lr=0.01)
criterion=nn.CrossEntropyLoss()

# Тренировка модели
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        opt.zero_grad() # Обнуление градиентов
        outputs = model(X_batch) # Запуск forward propagation
        loss = criterion(outputs, y_batch) # Считаем функцию потерь
        loss.backward()
        opt.step() # Обновляем веса

    # Проверка accuracy на тестах
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print(f"Epoch: {epoch+1}, Test accuracy: {correct/total:.3f}")

    # Сохраняем модель
    torch.save(model.state_dict(), "inference/isi_models.pth")
