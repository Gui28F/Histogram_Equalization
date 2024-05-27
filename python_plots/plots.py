import matplotlib.pyplot as plt

execution_times = {}

# Read the data from the file
with open("execution_times.txt", 'r') as file:
    for line in file:
        parts = line.strip().split(": ")
        thread_label = parts[0]
        times = list(map(float, parts[1].split(",")))
        execution_times[thread_label] = times



def plot_execution_times():
    seq_time = (sum(execution_times["1"]) / len(execution_times["1"]))
    # Calculate average execution time for each thread
    avg_times = {thread_label: seq_time/(sum(times) / len(times)) for thread_label, times in execution_times.items()}

    # Sort the dictionary by thread label
    avg_times_sorted = dict(sorted(avg_times.items(), key=lambda item: int(item[0][6:]) if item[0][6:].isdigit() else 0))


# Plot the data
    plt.plot(avg_times_sorted.keys(), avg_times_sorted.values(), marker='o', color='skyblue', linewidth=2, markersize=8)
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup for Each Number of Threads')
    plt.grid(True)
    plt.show()

"""
def box_plot():
    thread_times = [execution_times[thread_label] for thread_label in sorted(execution_times.keys(), key=lambda x: int(x))]

    # Create a box plot for each thread's execution times
    plt.boxplot(thread_times, labels=sorted(execution_times.keys(), key=lambda x: int(x)))
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Times for Each Number of Threads')
    plt.grid(True)
    plt.show()
"""
plot_execution_times()
#box_plot()