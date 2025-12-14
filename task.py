import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка датасета
df = pd.read_csv('dataset_simple.csv')

X = torch.Tensor(df.iloc[:, [0]].values)   # возраст
y = torch.Tensor(df.iloc[:, 1].values)     # доход

# Нормализация данных (по опыту л.р. по предмету "Машинное обучение" так результат будет лучше)
X_mean = X.mean()
X_std = X.std()

y_mean = y.mean()
y_std = y.std()

X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std


# Нейронная сеть для регрессии
class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, X):
        return self.layers(X)

# Параметры сети
inputSize = 1
hiddenSize = 8
outputSize = 1

net = NNet_regression(inputSize, hiddenSize, outputSize)

# Функция ошибки и оптимизатор
lossFn = nn.MSELoss()

# С оптимизатором Adam работает лучше, но можно оставить и SGD
#optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Обучение
epochs = 3500
for i in range(epochs):
    pred = net(X_norm).squeeze()
    loss = lossFn(pred, y_norm)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f'Эпоха {i}, ошибка: {loss.item():.2f}')

# Оценка результата
with torch.no_grad():
    pred_norm = net(X_norm).squeeze()
    
# Обратная нормализация    
pred = pred_norm * y_std + y_mean
y_real = y_norm * y_std + y_mean

mae = torch.mean(torch.abs(y_real - pred))
print('\nСредняя абсолютная ошибка (MAE):', mae.item())

# Визуализация
plt.figure()
plt.scatter(X.numpy(), y.numpy(), label='Реальные данные')
plt.scatter(X.numpy(), pred.numpy(), color='red', label='Предсказание сети')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.legend()
plt.show()
