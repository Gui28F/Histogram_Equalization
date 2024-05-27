import matplotlib.pyplot as plt
import os

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

def plot_execution_times(files):
    plt.figure(figsize=(10, 6))

    for file_name in files:
        execution_times = read_execution_times(file_name)
        speedup = calculate_speedup(execution_times)
        plt.plot(list(speedup.keys()), list(speedup.values()), marker='o', linewidth=2, markersize=8, label=os.path.basename(file_name.split(".")[0]))

    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup for Each Number of Threads')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage with a list of file names
files = ["borabora_1.txt", "input01.txt", "sample_5184×3456.txt"]
plot_execution_times(files)
