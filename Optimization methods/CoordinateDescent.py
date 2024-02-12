import numpy as np
import matplotlib.pyplot as plt

# Генерация рельефа (пример: яма)
def generate_terrain(x, y):
    return x ** 2 + y ** 2

# Расчет градиента по координатам
def calculate_gradient(func, point, step_size=0.09):
    gradient_x = func(point[0] + step_size, point[1]) - func(*point)
    gradient_y = func(point[0], point[1] + step_size) - func(*point)
    return np.array([gradient_x, gradient_y])

# Покоординатный спуск
def coordinate_descent(start_point, objective_function, step_size=0.09, max_iterations=1000):
    path = [start_point]
    current_position = start_point

    for _ in range(max_iterations):
        gradient = calculate_gradient(objective_function, current_position, step_size)

        if gradient.size != 2:
            raise ValueError("Функция должна возвращать градиент с двумя значениями")

        grad_x, grad_y = gradient

        move_direction = np.array([grad_x, 0]) if np.abs(grad_x) > np.abs(grad_y) else np.array([0, grad_y])
        next_position = current_position - move_direction

        path.append(next_position)
        current_position = next_position

    return np.array(path)

# Визуализация рельефа и траектории спуска
def plot_terrain_and_path(x_vals, y_vals, terrain_values, path, initial_position):
    plt.figure(figsize=(10, 8))
    plt.contourf(x_vals, y_vals, terrain_values, levels=50, cmap='viridis')
    plt.plot(path[:, 0], path[:, 1], 'r.-', label='Траектория спуска')
    plt.plot(initial_position[0], initial_position[1], 'bo', label='Начальная точка')
    plt.title('Покоординатный спуск лыжника')
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.colorbar(label='Высота')
    plt.legend()
    plt.show()

# Генерация сетки значений для рельефа
x_values = np.linspace(-10, 10, 400)
y_values = np.linspace(-10, 10, 400)
x_grid, y_grid = np.meshgrid(x_values, y_values)
terrain_values = generate_terrain(x_grid, y_grid)

# Начальная точка и выполнение покоординатного спуска
initial_position = np.array([8, 8])
descent_path = coordinate_descent(initial_position, generate_terrain)

# Визуализация
plot_terrain_and_path(x_grid, y_grid, terrain_values, descent_path, initial_position)
