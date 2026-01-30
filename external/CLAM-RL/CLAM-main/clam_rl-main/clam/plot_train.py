import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Function to extract data from a specific tag in event files
def extract_data_from_event_file(file_path, tag_name):
    ea = event_accumulator.EventAccumulator(file_path,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag_name in ea.Tags()['scalars']:
        events = ea.Scalars(tag_name)
        return [(e.wall_time, e.step, e.value) for e in events]
    else:
        print(f"Tag '{tag_name}' not found in {file_path}")
        return []

# Function to apply smoothing to the data
def smooth_data(data, weight=0.6):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# Function to plot data
def plot_data(data, smoothness=0.6, label=''):
    steps = [x[1] for x in data]
    values = [x[2] for x in data]

    if smoothness > 0:
        smoothed_values = smooth_data(values, smoothness)
    else:
        smoothed_values = values

    plt.plot(steps, smoothed_values, label=label)
    plt.xlabel('Steps (10 million)')
    plt.ylabel('Reward')
    plt.title('Comparison of Prey Average Reward')
    plt.legend()
    plt.grid(True)

# Main script to process and plot data from each subfolder
main_folder_path = 'results/clam_fnn_log/evaluates'
tag_of_interest = 'Prey average reward'
smoothness = 0.7

plt.figure(figsize=(12, 8))  # Initialize the plot

for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        aggregated_data = []
        for log_file in os.listdir(subfolder_path):
            if 'tfevents' in log_file:
                file_path = os.path.join(subfolder_path, log_file)
                data = extract_data_from_event_file(file_path, tag_of_interest)
                aggregated_data.extend(data)
        if aggregated_data:
            plot_data(aggregated_data, smoothness, label=subfolder)

plt.show()  # Show the plot after all data has been added
