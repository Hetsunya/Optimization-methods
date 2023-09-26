import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(c1, c2, x1, y1, x2, y2, x0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    x = x0[0]  # Инициализируем начальное значение x
    y = x0[1]  # Инициализируем начальное значение y на границе между берегом и водой
    x_history = [x]  # Для записи истории перемещения студента по оси x
    y_history = [y]  # Для записи истории перемещения студента по оси y
    iterations = 0  # Счетчик итераций

    while iterations < max_iterations:
        # Вычисляем градиент функции T относительно x
        gradient_x = (x1 - x) / (c1 * np.abs(x1 - x)) + (x - x2) / (c2 * np.sqrt((x - x2)**2 + y2**2))

        # Обновляем значение x с использованием градиентного спуска
        x -= learning_rate * gradient_x
        x_history.append(x)
        y_history.append(y)

        # Проверяем условие остановки
        if np.abs(gradient_x) < tolerance:
            break

        iterations += 1

    return x, y, x_history, y_history

# Исходные данные
c1 = 5.0  # Скорость студента на берегу
c2 = 3.0  # Скорость студента в воде
x1 = -10.0  # Координата студента на берегу (x1, y1)
y1 = -5.0  # Высота студента над водой (y1 < 0)
x2 = 8.0   # Координата девушки в воде (x2, y2)
y2 = 5.0   # Высота девушки над водой (y2 > 0)

# Начальное приближение (устанавливаем y=0 для границы между берегом и водой)
initial_guess = ((x1 + x2) / 2, 0)

# Решение задачи с помощью градиентного спуска
optimal_x, optimal_y, _, _ = gradient_descent(c1, c2, x1, y1, x2, y2, initial_guess)

# Графическое отображение передвижения студента
time_points = np.linspace(0, 1, 100)  # Генерируем равномерные точки времени от 0 до 1
x_values = [x1 + (optimal_x - x1) * t for t in time_points]  # Вычисляем соответствующие координаты x
y_values = [y1 + (optimal_y - y1) * t for t in time_points]  # Вычисляем соответствующие координаты y

# График
plt.figure(figsize=(10, 5))  # Задаем размер окна графика
plt.plot(x_values, y_values, 'bo-', label="Студент")
plt.plot([x2], [y2], 'go', label="Девушка")
plt.axhline(y=0, color='k', linestyle='--', label="Граница воды и берега")
plt.xlabel("Координата x")
plt.ylabel("Координата y")
plt.xlim(min(x1, x2) - 2, max(x1, x2) + 2)  # Задаем пределы осей для центрирования
plt.ylim(min(y1, 0) - 1, max(y2, 1) + 1)  # Задаем пределы осей для центрирования
plt.legend()
plt.title("Передвижение студента и девушки")
plt.grid(True)
plt.show()

print("Оптимальная координата (x):", optimal_x)
