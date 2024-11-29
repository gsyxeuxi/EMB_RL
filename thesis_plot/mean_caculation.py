import json
import numpy as np

# Load the JSON file
json_file_path = "./100test_log_1.json"
with open(json_file_path, 'r') as file:
    data = json.load(file)

# # Extract reward1, reward2, and reward3
# reward1_values = [entry["reward1"] for entry in data]
# reward2_values = [entry["reward2"] for entry in data]
# reward3_values = [entry["reward3"] for entry in data]

# # Calculate means
# reward1_mean = np.mean(reward1_values)
# reward2_mean = np.mean(reward2_values)
# reward3_mean = np.mean(reward3_values)

# # Calculate variances
# reward1_variance = np.var(reward1_values)
# reward2_variance = np.var(reward2_values)
# reward3_variance = np.var(reward3_values)

# # Print the results
# results = {
#     "Reward1": {"Mean": f"{reward1_mean:.2e}", "Variance": f"{reward1_variance:.2e}"},
#     "Reward2": {"Mean": f"{reward2_mean:.2e}", "Variance": f"{reward2_variance:.2e}"},
#     "Reward3": {"Mean": f"{reward3_mean:.2e}", "Variance": f"{reward3_variance:.2e}"},
# }

# print(results) # Display the calculated results

# Extract reward1, reward2, and reward3
reward = [entry["reward"] for entry in data]
fail = [entry["terminated"] for entry in data]

# Calculate means
reward1_mean = np.mean(reward)
fails = np.sum(fail)


# Calculate variances
reward_variance = np.var(reward)

# Print the results
results = {
    "Reward1": {"Mean": f"{reward1_mean:.2e}", "Variance": f"{reward_variance:.2e}"}, "Fail": f"{fails:.2e}"}

print(results) # Display the calculated results
