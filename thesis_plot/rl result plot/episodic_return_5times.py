# import json
# import numpy as np
# import matplotlib.pyplot as plt

# # List of JSON file paths
# file_paths = [
#     'EMB-fv-v11__ppo_fv_v11__643__20241016-train.json',
#     'EMB-fv-v11__ppo_fv_v11__1__20241016-train.json',
#     'EMB-fv-v11__ppo_fv_v11__35__20241018-train-1e-4lr.json',
#     'EMB-fv-v11__ppo_fv_v11__89__20241016-train.json',
#     'EMB-fv-v11__ppo_fv_v11__4911__20241027-train-good.json'
# ]

# # Initialize lists to collect training steps and episodic returns
# all_training_steps = []
# all_episodic_returns = []

# # Load data from each file
# for file_path in file_paths:
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#         training_steps = [entry[1] for entry in data]
#         episodic_returns = [entry[2] for entry in data]

#         all_training_steps.append(training_steps)
#         all_episodic_returns.append(episodic_returns)

# # Ensure all data has the same length
# min_length = min(len(steps) for steps in all_training_steps)
# all_training_steps = np.array([steps[:min_length] for steps in all_training_steps])
# all_episodic_returns = np.array([returns[:min_length] for returns in all_episodic_returns])

# # Calculate mean and standard error
# mean_returns = np.mean(all_episodic_returns, axis=0)
# std_returns = np.std(all_episodic_returns, axis=0)
# stderr_returns = std_returns / np.sqrt(len(file_paths))

# # Use the first file's training steps as the x-axis
# training_steps = all_training_steps[0]

# # Plot mean episodic return with error bars
# plt.figure(figsize=(10, 6))
# plt.plot(training_steps, mean_returns, label='Mean Episodic Return', linewidth=1.5)
# plt.fill_between(
#     training_steps,
#     mean_returns - stderr_returns,
#     mean_returns + stderr_returns,
#     alpha=0.3,
#     label='Std Error'
# )

# plt.xlabel('Training Steps', fontsize=12)
# plt.ylabel('Episodic Return', fontsize=12)
# plt.grid(True)

# # Customize x-axis ticks
# plt.ticklabel_format(style='scientific', axis='x', scilimits=(5, 5))

# plt.legend(fontsize=10)
# plt.tight_layout()

# # Save the figure
# plt.savefig("episodic_return_5_times_no.svg", format="svg")
# plt.show()

import json
import numpy as np
import matplotlib.pyplot as plt

# List of JSON file paths
file_paths = [
    'EMB-fv-v11__ppo_fv_v11__643__20241016-train.json',
    'EMB-fv-v11__ppo_fv_v11__1__20241016-train.json',
    'EMB-fv-v11__ppo_fv_v11__35__20241018-train-1e-4lr.json',
    'EMB-fv-v11__ppo_fv_v11__89__20241016-train.json',
    'EMB-fv-v11__ppo_fv_v11__4911__20241027-train-good.json'
]

# Initialize lists to collect training steps and episodic returns
all_training_steps = []
all_episodic_returns = []

# Load data from each file
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        training_steps = [entry[1] for entry in data]
        episodic_returns = [entry[2] for entry in data]

        all_training_steps.append(training_steps)
        all_episodic_returns.append(episodic_returns)

# Ensure all data has the same length
min_length = min(len(steps) for steps in all_training_steps)
all_training_steps = np.array([steps[:min_length] for steps in all_training_steps])
all_episodic_returns = np.array([returns[:min_length] for returns in all_episodic_returns])

# Calculate mean and standard error
mean_returns = np.mean(all_episodic_returns, axis=0)
std_returns = np.std(all_episodic_returns, axis=0)
stderr_returns = std_returns / np.sqrt(len(file_paths))

# Use the first file's training steps as the x-axis
training_steps = all_training_steps[0]

# Define a smoothing function
def smooth(data, window_size=5):
    """Apply a moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
window_size = 8  # Set the moving average window size
smoothed_mean_returns = smooth(mean_returns, window_size)
smoothed_upper = smooth(mean_returns + stderr_returns, window_size)
smoothed_lower = smooth(mean_returns - stderr_returns, window_size)

# Adjust training steps for the reduced length after smoothing
smoothed_training_steps = training_steps[:len(smoothed_mean_returns)]

# Plot smoothed mean episodic return with error bars
plt.figure(figsize=(10, 6))
plt.plot(smoothed_training_steps, smoothed_mean_returns, label='Smoothed Mean Episodic Return', linewidth=1.5)
plt.fill_between(
    smoothed_training_steps,
    smoothed_lower,
    smoothed_upper,
    alpha=0.3,
    label='Smoothed Std Error'
)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Episodic Return', fontsize=12)
plt.grid(True)

# Customize x-axis ticks
plt.ticklabel_format(style='scientific', axis='x', scilimits=(5, 5))

plt.legend(fontsize=10)
plt.tight_layout()

# Save the figure
plt.savefig("smoothed_episodic_return_5_times.svg", format="svg")
plt.show()
