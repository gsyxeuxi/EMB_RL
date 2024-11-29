import json
import numpy as np
import matplotlib.pyplot as plt
import os
print("Current working directory:", os.getcwd())

# Load the JSON data from two files
file_paths = [
    f'./EMB-fv-v6__PPO_fv_v6__1__test_of_fv_range.json',
    f'./EMB-fv-v7__PPO_fv_v7__1__test_of_fv_range.json'
]

# Colors and labels for the plots
colors = ['tab:blue', 'tab:orange']
labels = ['Sample-based parameter method', 'Fixed parameter method']

# Initialize lists to store data
training_steps_list = []
episodic_returns_list = []

# Read and process each JSON file
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        # training_steps = [entry[1] for entry in data]
        training_steps = [1e-5 + (5e-5 - 1e-5) * x / 39 for x in range(len(data))]
        episodic_returns = [entry[2] for entry in data]
        training_steps_list.append(training_steps)
        episodic_returns_list.append(episodic_returns)

# Plot the training results
plt.figure(figsize=(10, 6))

for training_steps, episodic_returns, color, label in zip(
        training_steps_list, episodic_returns_list, colors, labels):
    plt.plot(training_steps, episodic_returns, linewidth=1.5, color=color, label=label)

# Add labels, grid, and legend
plt.xlabel('Viscous Friction', fontsize=12)
plt.ylabel('Episodic Return', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)

# Set x-axis ticks with scientific notation
# plt.ticklabel_format(style='scientific', axis='x', scilimits=(5, 5))
plt.ticklabel_format(style='scientific', axis='x')


# Tight layout and save the figure
plt.tight_layout()
plt.savefig("./episodic_return_comparison.svg", format="svg")
plt.show()
