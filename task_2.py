import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Загрузка датасета
df = pd.read_csv('data.csv', skiprows=1, header=None)

# Разделение на признаки и метки класса 
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values

# Преобразование строковых меток класса в числовые
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Классы ирисов: {label_encoder.classes_}")
print(f"Количество классов для классификации: {num_classes}")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)

# Модель нейронной сети
class IrisClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IrisClassifier, self).__init__()
        # Первый полносвязный слой
        self.fc1 = nn.Linear(input_size, 10)
        # Функция активации ReLU
        self.relu = nn.ReLU()
        # Второй полносвязный слой
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        # Прямой проход данных через слои
        out = self.fc1(x)
        out = self.relu(out) 
        out = self.fc2(out)
        return out

# Параметры модели
input_size = X_train.shape[1] # Количество входных признаков
model = IrisClassifier(input_size, num_classes) # Экземпляр модели

# Функция потерь
criterion = nn.CrossEntropyLoss()

# Оптимизатор SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
num_epochs = 100 # Количество эпох
print(f"\nОбучение модели на {num_epochs} эпохах")

for epoch in range(num_epochs):
    # Обнуление градиента перед каждым новым проходом для предотвращения
    # накопления градиентов с предыдущих итераций
    optimizer.zero_grad()

    # Прямой проход (предсказания модели)
    outputs = model(X_train)

    # Вычисление ошибки (потери)
    loss = criterion(outputs, y_train)

    # Обратный проход, который вычислит градиенты функции потерь
    loss.backward()

    # Шаг оптимизации: обновление весов и смещения модели на основе градиентов
    optimizer.step()

    # Информация о потерях каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {loss.item():.4f}')

# Оценка модели на тестовой выборке
with torch.no_grad():
    outputs = model(X_test)

    # torch.max возвращает максимальное значение и его индекс.
    # Нужен индекс, который соответствует предсказанному классу.
    _, predicted = torch.max(outputs.data, 1)

    # Вычисление точности модели
    accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    print(f'\nТочность модели на тестовой выборке: {accuracy:.4f}')