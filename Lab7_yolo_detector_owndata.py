import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import ultralytics
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Проверка окружения
ultralytics.checks()

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")

# Проверка работы модели на одном изображении
test_image_path = "planes_trains_yolo/images/train/train_01.jpg"  

if os.path.exists(test_image_path):
    results = model(test_image_path)
    result = results[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(result.plot()[:, :, ::-1])
    plt.title("Результат детекции (до обучения)")
    plt.axis("off")
    plt.show()
else:
    print("Тестовое изображение не найдено")

# Обучение на собственном датасете
data_yaml_path = "data.yaml"

import os
print(os.getcwd())
print(os.listdir())

model.train(
    data=data_yaml_path,
    epochs=40,
    imgsz=640,
    batch=4,
    workers=0,
    verbose=True
)

# Проверка обученной модели на изображении
if os.path.exists(test_image_path):
    results = model(test_image_path)
    result = results[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(result.plot()[:, :, ::-1])
    plt.title("Результат детекции (после обучения)")
    plt.axis("off")
    plt.show()