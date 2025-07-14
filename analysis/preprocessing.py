import json
import numpy as np

def subtree_vals(path_value, choice1, tree_config=None):
    """
    For the first Solway tree configuration:
      - Left subtree has only one leaf (index 0)
      - Right subtree has two leaves (indices 1,2)
    """
    if tree_config == '1':
        if choice1 == 1:
            # left subtree: indices 0 and 1
            return path_value[0:2]
        else:
            # right subtree: index 2
            return [path_value[2]]
    else:
        if choice1 == 1:
            # left subtree: indices 0 and 1
            return path_value[0:2]
        else:
            # right subtree: indices 2 and 3
            return path_value[2:4]

def calculate_diff1(rewards):
    """Stage 1 difficulty: max reward minus mean of other rewards."""
    idx_max = np.argmax(rewards)
    others = [r for i,r in enumerate(rewards) if i != idx_max]
    return float(rewards[idx_max] - np.mean(others))

def calculate_diff2(value2, choice1, tree_config=None):
    """
    Stage 2 difficulty:
      - If chosen subtree has only one leaf, return -1
      - Otherwise, absolute difference between the two leaf values
    """
    vals = subtree_vals(value2, choice1, tree_config)
    if len(vals) < 2:
        return -1.0
    return float(abs(vals[0] - vals[1]))

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

def add_info_to_json(input_file, output_file, tree_config=None):
    # Read the original JSON file
    # with open(input_file, 'r') as f:
    #     trials = [json.loads(line) for line in f]

    with open(input_file) as f:
        trials = json.load(f)

    # Add difficulties to each trial
    for trial in trials:
        # Calculate stage 1 difficulty
        trial['diff1'] = round(calculate_diff1(trial['rewards']))
        
        # Calculate stage 2 difficulty
        trial['diff2'] = calculate_diff2(trial['value2'], trial['choice1'], tree_config)

        # Calculate overall trial difficulty from rewards
        trial['difficulty'] = calculate_diff1(trial['rewards'])
    
    # Group trials by participant
    participant_trials = {}
    for trial in trials:
        wid = trial['wid']
        if wid not in participant_trials:
            participant_trials[wid] = []
        participant_trials[wid].append(trial)
    
    # Process each participant's trials
    filtered_trials = []
    for wid, p_trials in participant_trials.items():
        
        # # Calculate RT statistics
        # rt1_mean = np.mean([t['rt1'] for t in p_trials])
        # rt1_std = np.std([t['rt1'] for t in p_trials])
        # rt1_threshold = rt1_mean + 2*rt1_std
        
        # rt2_mean = np.mean([t['rt2'] for t in p_trials])
        # rt2_std = np.std([t['rt2'] for t in p_trials])
        # rt2_threshold = rt2_mean + 2*rt2_std
        
        # rt_mean = np.mean([t['rt'] for t in p_trials])
        # rt_std = np.std([t['rt'] for t in p_trials])
        # rt_threshold = rt_mean + 2*rt_std
        
        # Count trials per difficulty level
        difficulty_counts = {}
        for trial in p_trials:
            diff = trial['difficulty']
            if diff not in difficulty_counts:
                difficulty_counts[diff] = 0
            difficulty_counts[diff] += 1

        # remove trials with rt1 < 500 or (rt1 or rt2) > 10000
        # p_trials = [t for t in p_trials if t['rt1'] >= 500 and (t['rt1'] <= 10000 and t['rt2'] <= 10000)]
        
        # Remove trials with less than 3 instances per difficulty
        # p_trials = [t for t in p_trials if difficulty_counts[t['difficulty']] > 5]
        # Filter trials
        # valid_trials = []
        # for trial in p_trials:
        #     if (trial['rt1'] <= rt1_threshold and 
        #         trial['rt2'] <= rt2_threshold and
        #         trial['rt'] <= rt_threshold):
        #         valid_trials.append(trial)
                
        # filtered_trials.extend(valid_trials)
        filtered_trials.extend(p_trials)

    # Filter out participants with too few trials
    participant_trial_counts = {}
    for trial in filtered_trials:
        wid = trial['wid']
        if wid not in participant_trial_counts:
            participant_trial_counts[wid] = 0
        participant_trial_counts[wid] += 1
    
    # Keep only trials from participants with >= 70 trials
    filtered_trials = [t for t in filtered_trials if participant_trial_counts[t['wid']] >= 70]
    
    print(f"Removed participants with fewer than 70 trials")
    print(f"Participants remaining: {len(set(t['wid'] for t in filtered_trials))}")
    
    # Update trials list
    trials_new = filtered_trials
    print(f"Removed trials with RT > 2 SD and less than 5 instances per difficulty")
    print(f"Removed {len(trials) - len(filtered_trials)} trials")
    
    # Convert numpy types to Python native types before JSON serialization
    trials_new = [convert_numpy_types(trial) for trial in trials_new]
    
    # Write the modified data to a new JSON file
    with open(output_file, 'w') as f:
        for trial in trials_new:
            json.dump(trial, f)
            f.write('\n')
    
    print(f"Added difficulties to {len(trials_new)} trials")
    print(f"Saved to {output_file}")

def main():
    input_file = 'data/simulation/simulate_model2.json'
    output_file = 'data/v3/simulated_v3_model2.json'
    add_info_to_json(input_file, output_file, tree_config='2')

if __name__ == "__main__":
    main() 