import json
import numpy as np

import numpy as np

# ===== Tree 6 constants =====
# Global leaf-path indices (left→right): [7, 8, 9, 10, 5, 6]
LEFT_PATH_IDX  = {0, 1, 2, 3}   # 7, 8, 9, 10
RIGHT_PATH_IDX = {4, 5}         # 5, 6

def _path_subtree(idx: int) -> int:
    """0 = left subtree, 1 = right subtree."""
    return 0 if idx in LEFT_PATH_IDX else 1

# ===== Difficulties =====

def calculate_diff1(path_rewards):
    """
    Stage 1 difficulty: (max path) - (mean of the others).
    """
    p = np.asarray(path_rewards, dtype=float)
    i_max = int(np.argmax(p))
    others = np.delete(p, i_max)
    return float(p[i_max] - np.mean(others))

def _subpaths_in_subtree(value2, value3, choice1):
    """
    Stage-2 subpaths in the chosen subtree (Tree 6):
      Left  (choice1=1): [3+7, 3+8, 4+9, 4+10]  (two double branches)
      Right (choice1=2): [5, 6]                 (two single branches)
    value2 = [3,4,5,6], value3 = [7,8,9,10] (both left→right).
    """
    v2 = np.asarray(value2, dtype=float)
    v3 = np.asarray(value3, dtype=float)
    if choice1 == 1:
        return np.array([v2[0] + v3[0], v2[0] + v3[1], v2[1] + v3[2], v2[1] + v3[3]])
    elif choice1 == 2:
        return np.array([v2[2], v2[3]])  # 5, 6
    else:
        return np.array([])

def calculate_diff2(value2, value3, choice1):
    """
    Stage 2 difficulty (generalized):
      difficulty = (max subpath) - (mean of all the other subpaths)
      Left  : max over [3+7, 3+8, 4+9, 4+10]
      Right : max over [5, 6]
    """
    sub = _subpaths_in_subtree(value2, value3, choice1)
    if sub.size < 2:
        return -1.0
    i_max = int(np.argmax(sub))
    others = np.delete(sub, i_max)
    return float(sub[i_max] - np.mean(others))

def calculate_diff3(value3, choice1, choice2):
    """
    Stage 3 difficulty:
      Only exists if Stage 2 chose a *double* branch (left, node3 or node4).
      Return |leafA - leafB| for that branch, else -1.0.
    """
    v3 = np.asarray(value3, dtype=float)  # [7,8,9,10]
    if choice1 == 1 and choice2 == 1:        # node 3 -> (7,8)
        return float(abs(v3[0] - v3[1]))
    if choice1 == 1 and choice2 == 2:        # node 4 -> (9,10)
        return float(abs(v3[2] - v3[3]))
    return -1.0

# ===== Correctness =====

def correct1(best_path_idx: int, choice1: int) -> bool:
    """
    Stage 1 correct if chosen subtree contains the global best path.
    """
    if choice1 == 1:
        return best_path_idx in LEFT_PATH_IDX
    if choice1 == 2:
        return best_path_idx in RIGHT_PATH_IDX
    return False

def correct2(value2, value3, choice1: int, choice2: int) -> bool:
    """
    Stage 2 correct if the chosen BRANCH attains the maximal subpath in that subtree.
      Left  : compare best(node3) vs best(node4)
              where best(node3)=max(3+7,3+8), best(node4)=max(4+9,4+10)
              choice2 in {1(node3), 2(node4)}
      Right : compare 5 vs 6, choice2 in {3(node5), 4(node6)}
    """
    v2 = np.asarray(value2, dtype=float)
    v3 = np.asarray(value3, dtype=float)

    if choice1 == 1:
        best_3 = max(v2[0] + v3[0], v2[0] + v3[1])  # node3
        best_4 = max(v2[1] + v3[2], v2[1] + v3[3])  # node4
        # which branch wins?
        best_branch = 1 if best_3 >= best_4 else 2
        return (choice2 == best_branch)

    elif choice1 == 2:
        best_branch = 3 if v2[2] >= v2[3] else 4  # 5 vs 6
        return (choice2 == best_branch)

    return False

def correct3(value3, choice1: int, choice2: int, choice3: int):
    """
    Stage 3 correct iff (left & picked a double branch at Stage 2) and
    the chosen leaf is the larger one in that branch.
      node3: compare 7 vs 8, choice3 in {1=7, 2=8}
      node4: compare 9 vs 10, choice3 in {5=9, 6=10}
    If Stage 2 picked right subtree (single branches), return None.
    """
    v3 = np.asarray(value3, dtype=float)
    if choice1 == 1 and choice2 == 1:
        best_leaf = 1 if v3[0] >= v3[1] else 2
        return (choice3 == best_leaf)
    if choice1 == 1 and choice2 == 2:
        best_leaf = 5 if v3[2] >= v3[3] else 6
        return (choice3 == best_leaf)
    return None

# ===== Accessors (for analysis/plots) =====

def subtree_vals(value2, value3, choice1, stage=2, choice2=None):
    """
    stage=2 -> return subpath values in the chosen subtree:
                 Left : [3+7, 3+8, 4+9, 4+10]
                 Right: [5, 6]
    stage=3 -> if a double branch is specified by `choice2`, return its two leaf values:
                 node3: [7, 8]
                 node4: [9, 10]
               else [] (e.g., right subtree or missing choice2)
    """
    v2 = np.asarray(value2, dtype=float)
    v3 = np.asarray(value3, dtype=float)
    if stage == 2:
        return _subpaths_in_subtree(v2, v3, choice1)
    elif stage == 3:
        if choice1 == 1 and choice2 == 1:
            return v3[0:2]     # 7,8
        if choice1 == 1 and choice2 == 2:
            return v3[2:4]     # 9,10
        return np.array([])
    else:
        raise ValueError("stage must be 2 or 3")

# ===== Relation code (for stratification) =====

def subtree_relation_code(path_rewards):
    """
    Encode where the TOP TWO full paths live and which *branch* the BEST uses:
      30/31: best = left-node3 (double),   second same/opposite subtree
      40/41: best = left-node4 (double),   second same/opposite subtree
      50/51: best = right-node5 (single),  second same/opposite subtree
      60/61: best = right-node6 (single),  second same/opposite subtree
    """
    p = np.asarray(path_rewards, dtype=float)
    order = np.argsort(-p)
    best, second = int(order[0]), int(order[1])

    # map best to a head code
    if best in (0, 1):        # 7 or 8  -> left-node3
        head = 30
    elif best in (2, 3):      # 9 or 10 -> left-node4
        head = 40
    elif best == 4:           # 5       -> right-node5
        head = 50
    else:                     # 6       -> right-node6
        head = 60

    tail = 0 if _path_subtree(best) == _path_subtree(second) else 1
    return head + tail


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
            if line:
                trials.append(json.loads(line))

    print(f"Loaded {len(trials)} trials")

    for trial in trials:
        # ---------- basics ----------
        trial['best_path_idx'] = int(np.argmax(trial['path_rewards']))

        # ---------- difficulties ----------
        diff1 = calculate_diff1(trial['path_rewards'])
        trial['diff1'] = round(diff1 * 2) / 2

        # Tree-6 Stage 2 needs value2, value3, choice1
        diff2 = calculate_diff2(trial['value2'], trial['value3'], trial['choice1'])
        trial['diff2'] = round(diff2 * 2) / 2

        # Stage 3 exists only when Stage 2 chose a double branch on the left
        if 'choice2' in trial:
            diff3_raw = calculate_diff3(trial['value3'], trial['choice1'], trial['choice2'])
        else:
            diff3_raw = -1.0
        trial['diff3'] = (round(diff3_raw * 2) / 2) if diff3_raw != -1.0 else -1.0

        # Overall difficulty (keep same definition as before)
        trial['difficulty'] = round(calculate_diff1(trial['path_rewards']) * 2) / 2

        # ---------- correctness ----------
        trial['correct1'] = int(correct1(trial['best_path_idx'], trial['choice1']))

        # Stage 2 correctness uses (value2, value3, choice1, choice2)
        trial['correct2'] = int(correct2(trial['value2'], trial['value3'],
                                         trial['choice1'], trial['choice2'])) \
                            if 'choice2' in trial else None

        # Stage 3 correctness (may be None if Stage 2 picked a single branch/right subtree)
        if 'choice3' in trial and 'choice2' in trial and trial['choice3'] is not None:
            c3 = correct3(trial['value3'], trial['choice1'], trial['choice2'], int(trial['choice3']))
            trial['correct3'] = (int(c3) if isinstance(c3, bool) else None)
        else:
            trial['correct3'] = None

        # Aggregate correctness
        if trial['correct2'] is None:
            trial['correct_all'] = int(bool(trial['correct1']))
        else:
            if trial['correct3'] is None:
                trial['correct_all'] = int(bool(trial['correct1']) and bool(trial['correct2']))
            else:
                trial['correct_all'] = int(bool(trial['correct1']) and bool(trial['correct2']) and bool(trial['correct3']))

        # ---------- relation code ----------
        trial['subtree_relation'] = subtree_relation_code(trial['path_rewards'])

        # ---------- RT transforms ----------
        # exponentiate if present; include rt3 if available
        for k in ('rt1', 'rt2', 'rt3', 'rt'):
            if k in trial and trial[k] is not None:
                trial[k] = int(np.exp(trial[k]))

    # keep trials whose overall difficulty is in [2, 10]
    trials_new = [t for t in trials if 2 <= t['difficulty'] <= 10]

    # convert numpy scalars/arrays to native types before dumping
    trials_new = [convert_numpy_types(t) for t in trials_new]

    with open(output_file, 'w') as f:
        for t in trials_new:
            json.dump(t, f)
            f.write('\n')

    print(f"Added Information to {len(trials_new)} trials")
    print(f"Removed {len(trials) - len(trials_new)} trials")
    print(f"Saved to {output_file}")

def main():
    input_file = 'Tree6/data/trial6_v7.json'
    output_file = 'Tree6/data/Tree6_v3.json'
    add_info_to_json(input_file, output_file) 

if __name__ == "__main__":
    main() 