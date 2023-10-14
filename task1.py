import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(c1, c2, x1, y1, x2, y2, x0, learning_rate=0.001, max_iterations=10000, tolerance=1e-6):
    x = x0[0]
    y = x0[1]
    x_history = [x]
    y_history = [y]
    crossing_boundary = False
    iterations = 0

    while iterations < max_iterations:
        if not crossing_boundary:
            gradient_x = (x1 - x) / (c1 * np.abs(x1 - x) + 1e-8)
        else:
            gradient_x = (x2 - x) / (c2 * np.abs(x2 - x) + 1e-8)

        x -= learning_rate * gradient_x
        x_history.append(x)
        y_history.append(y)

        if not crossing_boundary and y > 0:
            crossing_boundary = True

        if np.abs(gradient_x) < tolerance:
            break

        iterations += 1

    return x, y, x_history, y_history

c1 = 10.0  # Скорость студента на берегу
c2 = 1.0   # Скорость студента в воде
x1 = -10.0
y1 = -5.0
x2 = 8.0
y2 = 5.0

# Начальное приближение внутри воды
initial_guess = ((x1 + x2) / 2, 0)

optimal_x, optimal_y, x_history, y_history = gradient_descent(c1, c2, x1, y1, x2, y2, initial_guess)

# Графическое отображение передвижения студента
plt.figure(figsize=(7, 5))
plt.plot(x_history, y_history, 'bo-', label="Студент")
plt.plot([x2], [y2], 'go', label="Девушка (фиксировано)")
plt.axhline(y=0, color='k', linestyle='--', label="Граница воды и берега")
plt.xlabel("Координата x")
plt.ylabel("Координата y")
plt.xlim(min(x1, x2) - 2, max(x1, x2) + 2)
plt.ylim(min(y1, 0) - 1, max(y2, 1) + 1)

# Добавляем линию для границы
plt.plot([optimal_x, optimal_x], [min(y1, 0) - 1, max(y2, 1) + 1], 'r--', label="Граница")

plt.legend()
plt.grid(True)
plt.show()

print("Оптимальная координата (x):", optimal_x)
print("Оптимальная координата (y):", optimal_y)