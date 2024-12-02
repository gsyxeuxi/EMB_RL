# import json
# import numpy as np

# # Load the JSON file
# json_file_path1 = "./100test_log_1.json"
# json_file_path2 = "./100test_log.json"

# with open(json_file_path, 'r') as file:
#     data = json.load(file)

# # Extract reward1, reward2, and reward3
# reward = [entry["reward"] for entry in data]
# fail = [entry["terminated"] for entry in data]

# # Calculate means
# reward1_mean = np.mean(reward)
# fails = np.sum(fail)


# # Calculate variances
# reward_variance = np.var(reward)

# # Print the results
# results = {
#     "Reward1": {"Mean": f"{reward1_mean:.2e}", "Variance": f"{reward_variance:.2e}"}, "Fail": f"{fails:.2e}"}

# print(results) # Display the calculated results

import json
import numpy as np

# Load the JSON files
json_file_path1 = "./100test_log_1.json"
json_file_path2 = "./100test_log.json"

with open(json_file_path1, 'r') as file1:
    data1 = json.load(file1)

with open(json_file_path2, 'r') as file2:
    data2 = json.load(file2)

# Extract reward and terminated status from file 1
reward1 = [entry["reward"] for entry in data1]
fail1 = [entry["terminated"] for entry in data1]

# Filter rewards for non-fail entries in file 1
non_fail_indices = [i for i, entry in enumerate(data1) if not entry["terminated"]]
non_fail_rewards1 = [reward1[i] for i in non_fail_indices]

# Extract corresponding rewards from file 2
non_fail_rewards2 = [data2[i]["reward"] for i in non_fail_indices]

# Calculate means
reward_mean_non_fail1 = np.mean(non_fail_rewards1)
reward_mean_non_fail2 = np.mean(non_fail_rewards2)
reward1_mean = np.mean(reward1)
fails = np.sum(fail1)

# Calculate variances
reward_variance = np.var(reward1)

# Print the results
results = {
    "Reward1": {
        "Mean": f"{reward1_mean:.2e}",
        "Variance": f"{reward_variance:.2e}"
    },
    "Fail": f"{fails:.2e}",
    "Reward1_Non_Fail_Mean": f"{reward_mean_non_fail1:.2e}",
    "Reward2_Non_Fail_Mean": f"{reward_mean_non_fail2:.2e}"
}

print(results) # Display the calculated results