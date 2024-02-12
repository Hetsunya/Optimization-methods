import matplotlib.pyplot as plt

def calculate_trajectory(y, delta_t, function, target_height, total_time):
    times = [0]
    heights = [y]

    for t in range(1, total_time + 1):
        y = y + delta_t * function(y, target_height)
        times.append(t)
        heights.append(y)

    return times, heights

def flight_function(y, target_height):
    # Пример функции полета
    return -(y - target_height) / 10

def plot_trajectory(times, heights):
    plt.plot(times, heights, label='Траектория полета')
    plt.xlabel('Время')
    plt.ylabel('Высота')
    plt.title('Траектория полета самолета')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    initial_height = 10 # начальная высота
    delta_t = 1        # шаг по времени
    total_time = 1000      # общее время моделирования

    target_height = 15   # целевая высота

    times, heights = calculate_trajectory(initial_height, delta_t, flight_function, target_height, total_time)
    plot_trajectory(times, heights)
