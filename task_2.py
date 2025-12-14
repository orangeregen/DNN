import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# загрузка датасета
df = pd.read_csv('data.csv', skiprows=1, header=None)

X = torch.Tensor(df.iloc[:, 0:3].values)
y_raw = df.iloc[:, 4].values

# бинарная кодировка классов
y = torch.Tensor(np.where(y_raw == 'Iris-setosa', 1, -1)).reshape(-1, 1)

# Линейный классификатор
model = nn.Linear(X.shape[1], 1)

# функция ошибки и оптимизатор
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Ошибка до обучения
with torch.no_grad():
    pred = model(X)
    pred_lbl = torch.where(pred >= 0, 1, -1)
    err = torch.sum(torch.abs(y - pred_lbl)) / 2
    print('Ошибок до обучения:', err.item())

# Обучение
epochs = 100
for i in range(epochs):
    pred = model(X)
    loss = lossFn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f'Эпоха {i}, ошибка MSE: {loss.item():.4f}')

# Ошибка после обучения
with torch.no_grad():
    pred = model(X)
    pred_lbl = torch.where(pred >= 0, 1, -1)
    err = torch.sum(torch.abs(y - pred_lbl)) / 2
    print('\nОшибок после обучения:', err.item())
