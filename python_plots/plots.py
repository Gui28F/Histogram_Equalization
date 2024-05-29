import matplotlib.pyplot as plt
import os
import numpy as np
gpu_ex = [76,63,2635]

def read_execution_times(file_name):
    execution_times = {}
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split(": ")
            thread_label = parts[0]
            times = list(map(float, parts[1].split(",")))
            execution_times[thread_label] = times
    return execution_times

def calculate_speedup(execution_times):
    seq_time = sum(execution_times["1"]) / len(execution_times["1"])
    avg_times = {thread_label: seq_time / (sum(times) / len(times)) for thread_label, times in execution_times.items()}
    avg_times_sorted = dict(sorted(avg_times.items(), key=lambda item: int(item[0][6:]) if item[0][6:].isdigit() else 0))
    return avg_times_sorted

def plot_execution_times(files, gpu_ex):
    plt.figure(figsize=(10, 6))

    for idx, file_name in enumerate(files):
        execution_times = read_execution_times(file_name)
        speedup = calculate_speedup(execution_times)

        # Plot CPU speedup and capture the line color
        line, = plt.plot(list(speedup.keys()), list(speedup.values()), marker='o', linewidth=2, markersize=8, label=os.path.basename(file_name.split(".")[0]))
        color = line.get_color()

        # Calculate and plot GPU speedup line with the same color
        seq_time = sum(execution_times["1"]) / len(execution_times["1"])
        gpu_speedup = seq_time / gpu_ex[idx]
        print(gpu_speedup)
        plt.axhline(y=gpu_speedup, color=color, linestyle='--', label=f'GPU {os.path.basename(file_name.split(".")[0])}')

    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup for Each Number of Threads (Iterations = 100)')
    plt.legend()
    plt.grid(True)
    plt.show()

def a ():
    import matplotlib.pyplot as plt

    tile_width = [8, 14, 16, 18, 20, 24, 32]
    time_borabora_1 = [315, 306, 280, 300, 281, 281, 292]
    time_input01 = [95, 91, 80, 89, 90, 84, 84]
    time_sample = [3225, 3080, 2600, 3000, 2865, 2867, 2695]


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(tile_width, time_borabora_1, marker='o', label='borabora_1')
    plt.plot(tile_width, time_input01, marker='o', label='input01')
    plt.plot(tile_width, time_sample, marker='o', label='sample_5184×3456')

    # Labels and Title
    plt.xlabel('Tile Width')
    plt.ylabel('Time')
    plt.xticks(tile_width, None)
    plt.title('Time vs Tile Width for Different Datasets')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()
#a()
# Example usage with a list of file names
files = ["borabora_1.txt", "input01.txt" ,"sample_5184×3456.txt"]
plot_execution_times(files, gpu_ex)