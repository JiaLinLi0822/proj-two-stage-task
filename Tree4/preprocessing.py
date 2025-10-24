import json
import numpy as np

def calculate_diff1(path):
    """
    Calculate stage 1 difficulty:
    max path reward minus mean of other path rewards.
    """
    path = np.array(path)
    idx_max = np.argmax(path)
    others = np.delete(path, idx_max)
    return path[idx_max] - np.mean(others)


def calculate_diff2(v2, choice1):
    """
    Calculate stage 2 difficulty for Tree1.
    If chosen subtree has only one leaf, return -1.
    Otherwise, return absolute difference between two leaf values.
    """
    vals = subtree_vals(v2, choice1)
    if len(vals) < 2:
        return -1.0
    return abs(vals[0] - vals[1])


def subtree_vals(v2, choice1):
    """
    Return leaf values in the chosen subtree.
    - Left subtree (choice1 == 1): indices 0,1
    - Middle subtree (choice1 == 2): indices 2,3
    - Right subtree (choice1 == 3): indices 4,5
    """
    if choice1 == 1:
        return v2[0:2]
    elif choice1 == 2:
        return v2[2:4]
    else:
        return v2[4:6]

def correct1(best_path_idx, choice1):
    return ((best_path_idx in [0, 1]) and (choice1 == 1) or
            (best_path_idx in [2, 3]) and (choice1 == 2) or
            (best_path_idx in [4, 5]) and (choice1 == 3))


def correct2(v2, choice1, choice2):
    vals = subtree_vals(v2, choice1)
    if choice1 == 1:
        local = choice2 - 1
        return vals[local] == max(vals)
    elif choice1 == 2:
        local = choice2 - 3
        return vals[local] == max(vals)
    elif choice1 == 3:
        local = choice2 - 5
        return vals[local] == max(vals)


def subtree_relation_code(path):
    """
    Relation between best path and other path locations.
    Return:
    1 = best in right subtree
    2..5 = best and kth-best in same subtree
    """
    path = np.array(path)
    # sortperm in descending order: indices of sorted elements
    idx_desc = np.argsort(-path)  # negative for descending
    best, second, third, fourth, fifth, worst = idx_desc

    def subtree(i):
        # left=0, middle=1, right=2
        if i <= 1:
            return 0
        elif i <= 3:
            return 1
        elif i <= 5:
            return 2

    if subtree(best) == subtree(second):
        return 1
    elif subtree(best) == subtree(third):
        return 2
    elif subtree(best) == subtree(fourth):
        return 3
    elif subtree(best) == subtree(fifth):
        return 4
    elif subtree(best) == subtree(worst):
        return 5


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def add_info_to_json(input_file, output_file):

    trials = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                trials.append(json.loads(line))

    print(f"Loaded {len(trials)} trials")
    # Add difficulties to each trial
    for trial in trials:

        trial['best_path_idx'] = np.argmax(trial['path_rewards'])
        # Calculate stage 1 difficulty
        trial['diff1'] = round(calculate_diff1(trial['path_rewards']) * 2) / 2
        # trial['diff1'] = round(calculate_diff1(trial['path_rewards']))
        
        # Calculate stage 2 difficulty
        trial['diff2'] = round(calculate_diff2(trial['value2'], trial['choice1']) * 2) / 2
        # trial['diff2'] = round(calculate_diff2(trial['value2'], trial['choice1']))

        # Calculate overall trial difficulty from path
        trial['difficulty'] = round(calculate_diff1(trial['path_rewards']) * 2) / 2
        # trial['difficulty'] = round(calculate_diff1(trial['path_rewards']))

        # Calculate correct1
        trial['correct1'] = int(correct1(trial['best_path_idx'], trial['choice1']))

        # Calculate correct2
        trial['correct2'] = int(correct2(trial['value2'], trial['choice1'], trial['choice2']))

        # Calculate correct_all
        trial['correct_all'] = int(trial['correct1'] and trial['correct2'])

        trial['subtree_relation'] = subtree_relation_code(trial['path_rewards'])


        trial['rt1'] = int(np.exp(trial['rt1']))
        trial['rt2'] = int(np.exp(trial['rt2']))
        trial['rt'] = int(np.exp(trial['rt']))
    
    # Only keep the trials with difficulty between 2 and 10
    trials_new = [trial for trial in trials if trial['difficulty'] <= 10 and trial['difficulty'] >= 2]

    # Convert numpy types to Python native types before JSON serialization
    trials_new = [convert_numpy_types(trial) for trial in trials_new]
    
    # Write the modified data to a new JSON file
    with open(output_file, 'w') as f:
        for trial in trials_new:
            json.dump(trial, f)
            f.write('\n')
    
    print(f"Added Information to {len(trials_new)} trials")
    print(f"Removed {len(trials) - len(trials_new)} trials")
    print(f"Saved to {output_file}")

def main():
    input_file = 'Tree4/data/trial4_v7.json'
    output_file = 'Tree4/data/Tree4_v3.json'
    add_info_to_json(input_file, output_file) 

if __name__ == "__main__":
    main() 