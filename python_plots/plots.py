import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_execution_times(file_path):
    num_iterations = []
    execution_times = []
    seq_time = None
    # Read the data from the file
    with open(file_path, 'r') as file:
        for line in file:
            iteration, time_ms = line.split(", ")
            num_iterations.append(int(iteration))
            if seq_time is not None:
                execution_times.append(seq_time / float(time_ms))
            else:
                seq_time = float(time_ms)
                execution_times.append(1)
    # Plot the data
    plt.plot(num_iterations, execution_times, marker='o')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Threads')
    plt.grid(True)
    plt.show()


# Example usage:
plot_execution_times("execution_times.txt")
