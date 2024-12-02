# import json
# import numpy as np
# import matplotlib.pyplot as plt


# labels = ["Action", "Reward", "Position", "Velocity"]

# # Initialize lists to collect training steps and episodic returns
# all_times = []
# all_values = []
# # List of JSON file paths
# file_paths = [
#     './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-act.json',
#     './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-reward.json',
#     './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-pos.json',
#     './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-vel.json'
# ]

# # Load data from each file
# for file_path in file_paths:
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#         times = [entry[1] for entry in data]
#         values = [entry[2] for entry in data]
#         all_times.append(times)
#         all_values.append(values)

# # Plotting the data in a (4, 1) format
# fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# for i, ax in enumerate(axes):
#     ax.plot(all_times[i], all_values[i], label=labels[i])
#     ax.set_title(labels[i])
#     if i == 0:
#         ax.set_ylabel('Input Voltage (V)')
#     if i == 1:
#         ax.set_ylabel('Step Reward')
#     if i == 2:
#         ax.set_ylabel('Motor Position (rad)')
#     if i == 3:
#         ax.set_ylabel('Motor Velocity (rad/s)')
#     ax.grid(True)
#     ax.set_xlabel('Time (ms)')

# plt.grid(True)

# plt.legend(fontsize=10)
# plt.tight_layout()

# # Save the figure
# plt.savefig("smoothed_episodic_return_5_times.svg", format="svg")
# plt.show()


import json
import numpy as np
import matplotlib.pyplot as plt

# Labels for the plots
labels = ["Action", "Reward", "Position", "Velocity"]
groups = ["Manually", "Method 1", "Method 2"]

# JSON file paths for three groups
file_paths_grouped = [
    [
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-act.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-reward.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-pos.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-manual-test-vel.json'
    ],
    [
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v1-test-act.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v1-test-reward.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v1-test-pos.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v1-test-vel.json'
    ],
    [
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v2-test-act.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v2-test-reward.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v2-test-pos.json',
        './EMB-fvk1-v1__ppo_fvk1_v1__1__20241127-v2-test-vel.json'
    ]
]

# Colors for each group
colors = ['r', 'g', 'b']

# Initialize the plot
fig, axes = plt.subplots(4, 1, figsize=(12, 15))

# Loop through each group and plot
for group_idx, group_paths in enumerate(file_paths_grouped):
    for file_idx, file_path in enumerate(group_paths):
        # Load data from JSON files
        with open(file_path, 'r') as file:
            data = json.load(file)
            times = [entry[1] for entry in data]
            values = [entry[2] for entry in data]
            
        # Plot each data set on the corresponding subplot
        axes[file_idx].plot(times, values, label=f'{groups[group_idx]}', color=colors[group_idx])
        # axes[file_idx].set_title(labels[file_idx])
        axes[file_idx].set_ylabel(
            ["Input Current (A)", "Step Reward", "Motor Position (rad)", "Motor Velocity (rad/s)"][file_idx]
        )
        axes[file_idx].grid(True)

# Set the xlabel and add legends to all subplots
for ax in axes:
    ax.set_xlabel('Time (ms)')
    ax.legend(fontsize=10)

# Tight layout and save the figure
plt.tight_layout()
plt.savefig("multi_group_episodic_return.svg", format="svg")
plt.show()
