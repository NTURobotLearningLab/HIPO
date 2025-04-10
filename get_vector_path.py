import os
import sys
import pickle

def get_vector(root_path, vector_file_name):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if vector_file_name in filenames:
            matches.append(os.path.join(dirpath, vector_file_name))
    return matches

root_path = sys.argv[1]
vector_file_name = sys.argv[2]
results = get_vector(root_path, vector_file_name)

max_reward = -1
vector_path = None
for temp_path in results:
    try:
        with open(temp_path, 'rb') as f:
            file = pickle.load(f)
    except Exception as e:
        print(e)
        continue

    if isinstance(file, dict):
        reward = file['actual_reward']

    elif isinstance(file, list):
        reward = file[0]['actual_reward']

    if reward >= max_reward:
        max_reward = reward
        vector_path = temp_path

print(vector_path)
