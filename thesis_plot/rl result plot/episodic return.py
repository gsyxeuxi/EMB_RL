import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
file_path = f'rl result plot/EMB-fv-v7__PPO_fv_v7__1__train.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract training steps and episodic returns
training_steps = [entry[1] for entry in data]

episodic_returns = [entry[2] for entry in data]

# Plot the training results
plt.figure(figsize=(10, 4))
plt.plot(training_steps, episodic_returns, linewidth=1.5)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Episodic Return', fontsize=12)
# plt.title('Training Progress Over Time')
plt.grid(True)
# Set x-axis ticks at intervals of 0.1M (100k) steps
# x_ticks = np.linspace(0, 1e6, 11) 
# plt.xticks(x_ticks, labels=[f'{int(x/1e5)}' for x in x_ticks])

plt.ticklabel_format(style='scientific', axis='x', scilimits=(5, 5))

plt.tight_layout()

plt.savefig("episodic return.svg", format="svg")
