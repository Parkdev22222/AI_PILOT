import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Revised function to safely convert string to list/tuple
def safe_eval_revised(val):
    try:
        val = val.strip('[]')
        val_list = val.split()
        if len(val_list) == 2:  # Ensure there are two elements
            return tuple(map(float, val_list))
        return []
    except:
        return []

# Function to find the longest sequences within trails
def find_longest_sequences_within_trails(series, trail_length):
    longest_sequences = []

    for start in range(0, len(series), trail_length):
        trail_end = start + trail_length
        trail_series = series[start:trail_end]

        last_value = None
        current_length = 0
        max_length = 0
        max_value = None

        for value in trail_series:
            if value == last_value:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    max_value = last_value
                last_value = value
                current_length = 1

        if current_length > max_length:
            max_length = current_length
            max_value = last_value

        if max_length > 0:
            longest_sequences.append((max_value, start, max_length))

    return sorted(longest_sequences, key=lambda x: x[2], reverse=True)[:5]

# Load the data
file_path = 'clam/all_data_policy_10.csv'  # Adjust the file path as needed
data = pd.read_csv(file_path)

# Apply the revised safe_eval function
for col in ['predator_1', 'predator_2', 'predator_3', 'prey_1']:
    data[col] = data[col].apply(safe_eval_revised)

# Find the top 5 longest sequences within each trail
top_5_sequences_within_trails = find_longest_sequences_within_trails(data['fss_argmax'], 50)

# Load the .npz file
npz_file_path = 'clam/policy10_image.npz'  # Replace with the path to your .npz file
npz_data = np.load(npz_file_path)
# Define the frame size
frame_size = 700
def safe_eval_cq_outs(val):
    try:
        return ast.literal_eval(val)
    except:
        return []
def transform_coords(coords):
    return [(x + 1) * frame_size / 2 for x in coords]
# Define specific colors for predators and prey
# ...

# Define specific colors for predators and prey
colors = {
    'predator_1': 'darkred',  # Different shades of red for predators
    'predator_2': 'red',
    'predator_3': 'lightcoral',
    'prey_1': 'green'  # Green for prey
}

custom_labels = {
    'predator_1': 'Predator 1',
    'predator_2': 'Predator 2',
    'predator_3': 'Predator 3',
    'prey_1': 'Prey'
}

latex_table = "\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n\\hline\n"
latex_table += "Rule ID & FSS & CQ Outs 1 & CQ Outs 2 & CQ Outs 3 & CQ Outs 4 & CQ Outs 5 \\\\\n\\hline\n"

# Plot the trajectories for the top 5 sequences
for value, start, length in top_5_sequences_within_trails:
    print(f"Value: {value}, Trail Start Index: {start}, Length: {length}")

    start_index = start
    trajectories = {
        'predator_1': data['predator_1'][start_index+int(length/2)-5:start_index + int(length/2)+5],
        'predator_2': data['predator_2'][start_index+int(length/2)-5:start_index + int(length/2)+5],
        'predator_3': data['predator_3'][start_index+int(length/2)-5:start_index + int(length/2)+5],
        'prey_1': data['prey_1'][start_index+int(length/2)-5:start_index + int(length/2)+5]
    }

    plt.figure(figsize=(10, 10))

    # Last Frame as Background
    last_frame_index = start_index + int(length/2)+5 - 1
    # print()
    last_frame_image = npz_data['arr_{}'.format(last_frame_index)]
    print("last frame", last_frame_index)
    plt.imshow(last_frame_image, extent=[0, frame_size, 0, frame_size])
    end_index = last_frame_index
    policy_id = data['policy_id'].iloc[end_index]
    rule_id = data['fss_argmax'].iloc[end_index]
    fss_values = data['fss'].iloc[end_index]  # Assuming this is already a list
    cq_outs = data['cq_outs'].iloc[end_index]  # Assuming this needs to be converted from a string
    fss_values = ast.literal_eval(fss_values)
    cq_outs = ast.literal_eval(cq_outs)  # Convert to list

    # Start LaTeX table
    latex_table = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
    latex_table += "Rule ID & Firing Strength & Consequent 1& Consequent 2 & Consequent 3 & Consequent 4 & Consequent 5 \\\\\n\\hline\n"

    # Add rows for each FSS value
    for fss_value, cq_row in zip(fss_values, cq_outs):
        formatted_row = [rule_id] + [round(fss_value, 2)] + [round(item, 2) for item in cq_row]
        latex_table += " & ".join(map(str, formatted_row)) + " \\\\\n\\hline\n"

    # End LaTeX table
    latex_table += "\\end{tabular}"

    # Print the LaTeX table
    print(latex_table)

    # First Frame Overlay with Reduced Opacity
    first_frame_image = npz_data['arr_{}'.format(start_index+int(length/2)-5)]
    print("first frame",start_index+int(length/2)-5)
    plt.imshow(first_frame_image, extent=[0, frame_size, 0, frame_size], alpha=0.3)

    # Plotting Trajectories, Arrows, and Points
    for agent, coords in trajectories.items():


        scaled_coords = [transform_coords(coord) for coord in coords]
        x, y = zip(*scaled_coords)

        # Add arrows for movement direction
        for i in range(len(x) - 1):
            plt.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i], color=colors[agent],
                      shape='full', lw=2, length_includes_head=True, head_width=10)

        # Highlight the start and end points
        plt.plot(x[0], y[0], color=colors[agent], marker='^', markersize=10)  # Start point with triangle marker
        plt.plot(x[-1], y[-1], color=colors[agent], marker='o', markersize=10)  # End point with circle marker

        # Label only once for each agent
        if start_index == start:
            plt.plot([], [], color=colors[agent], label=custom_labels[agent], marker='o', markersize=10)

    plt.title(f"Trajectory of Combination: {int(policy_id)+1}, Length: {length}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()

    plt.show()

