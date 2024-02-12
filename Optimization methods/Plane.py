from matplotlib import pyplot as plt
import math

def altitude_decision(current_height, desired_height):
    return 1 if current_height <= desired_height else -1

def simulate_flight(initial_height, launch_angle, target_height, total_flight_time, time_step):
    current_height = initial_height
    time_points = [i * time_step for i in range(int(total_flight_time // time_step) + 1)]
    height_points = []

    for t in time_points:
        height_points.append(current_height)
        delta_height = time_step * math.tan(math.radians(launch_angle)) * altitude_decision(current_height, target_height)
        current_height += delta_height

    return time_points, height_points

def plot_flight_simulations(initial_conditions, launch_angle, target_height, total_flight_time, time_step):
    plt.figure(figsize=(12, 6))
    plt.axhline(y=target_height, color='r', linestyle='--', label="Желаемая высота")
    plt.title("Моделирование системы автоматического контроля высоты с обновленными исходными условиями")
    plt.xlabel('Time (t)')
    plt.ylabel('Height (y(t))')
    plt.legend()
    plt.grid(True)

    for initial_height in initial_conditions:
        time_points, height_points = simulate_flight(initial_height, launch_angle, target_height, total_flight_time, time_step)
        plt.plot(time_points, height_points, label=f"Начало = {initial_height}")

    plt.show()

if __name__ == "__main__":
    target_height = 1
    total_flight_time = 45
    launch_angle = 45
    time_step = 0.1
    initial_conditions = [target_height + 10, target_height - 10, target_height]

    plot_flight_simulations(initial_conditions, launch_angle, target_height, total_flight_time, time_step)
