import torch 
import random

########### Задание 1 ###########
# тензор x целочисленного типа, хранящий случайное значение

x = torch.randint(0, 20, (1,),dtype=torch.int32)
print(f"Тензор x: {x}")

# преобразование к типу float32
x = x.to(dtype=torch.float32)
print(f"Тензор x в float32: {x}")

x.requires_grad=True

# возведение в степень n = 3
n = 3
x_n = x ** n
print(f"Тензор x в степени {n}: {x_n}")

# умножение на случайное значение в диапазоне от 1 до 10
rand_mult = random.uniform(1, 10)
random_multiplier = round(rand_mult, 2)
x_mult = x_n * random_multiplier
print(f"Умножение на случайное значение ({random_multiplier:.2f}): {x_mult}")

# взятие экспоненты от полученного числа
x_exp = torch.exp(x_mult)
print(f"Взятие экспоненты: {x_exp}")

# Значение производной для полученного в п.3 значения по x.
x_exp.backward()
print(f"Значение производной d(x_exp)/dx: {x.grad}")
