import json
import numpy as np

LEFT_PATH_IDX  = {0, 1, 2}   # 7, 8, 4
RIGHT_PATH_IDX = {3, 4, 5}   # 5, 9, 10

def _path_subtree(idx: int) -> int:
    """0 = left subtree, 1 = right subtree (by global leaf-path index 0..5)."""
    return 0 if idx in LEFT_PATH_IDX else 1

def calculate_diff1(path_rewards):
    """
    Stage 1 difficulty: (max path reward) - (mean of the other paths).
    """
    p = np.asarray(path_rewards, dtype=float)
    i_max = int(np.argmax(p))
    others = np.delete(p, i_max)
    return float(p[i_max] - np.mean(others))

def _subpaths_in_subtree(value2, value3, choice1):
    """
    Return the THREE subpath values in the chosen subtree for Stage 2:
      Left  (choice1=1): [3+7, 3+8, 4]
      Right (choice1=2): [6+9, 6+10, 5]
    value2 = [3,4,5,6], value3 = [7,8,9,10] (both left->right).
    """
    v2 = np.asarray(value2, dtype=float)   # [3,4,5,6]
    v3 = np.asarray(value3, dtype=float)   # [7,8,9,10]
    if choice1 == 1:
        return np.array([v2[0] + v3[0], v2[0] + v3[1], v2[1]])     # 3+7, 3+8, 4
    elif choice1 == 2:
        return np.array([v2[3] + v3[2], v2[3] + v3[3], v2[2]])     # 6+9, 6+10, 5
    else:
        return np.array([])

def calculate_diff2(value2, value3, choice1):
    """
    Stage 2 difficulty (your new rule):
      In the chosen subtree, compute the three subpaths:
        Left : [3+7, 3+8, 4]
        Right: [6+9, 6+10, 5]
      difficulty = (max of three) - (mean of the other two).
    """
    sub = _subpaths_in_subtree(value2, value3, choice1)
    if sub.size != 3:
        return -1.0
    i_max = int(np.argmax(sub))
    others = np.delete(sub, i_max)
    return float(sub[i_max] - np.mean(others))

def calculate_diff3(value3, choice1, choice2):
    """
    Stage 3 difficulty:
      If Stage 2 chose the double-leaf branch (node 3 on left, node 6 on right),
      return |leafA - leafB| for that branch; else return -1.0.
    value3 = [7,8,9,10].
    """
    v3 = np.asarray(value3, dtype=float)
    if choice1 == 1 and choice2 == 1:       # node 3 branch -> leaves 7,8
        return float(abs(v3[0] - v3[1]))
    if choice1 == 2 and choice2 == 4:       # node 6 branch -> leaves 9,10
        return float(abs(v3[2] - v3[3]))
    return -1.0

def correct1(best_path_idx: int, choice1: int) -> bool:
    """
    Stage 1 correct if the chosen subtree contains the global best path.
    best_path_idx in [0..5] for paths [7,8,4,5,9,10]; choice1: 1=left, 2=right.
    """
    if choice1 == 1:
        return best_path_idx in LEFT_PATH_IDX
    if choice1 == 2:
        return best_path_idx in RIGHT_PATH_IDX
    return False


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

def correct1(best_path_idx: int, choice1: int) -> bool:
    """
    Stage 1 correct if the chosen subtree contains the global best path.
    best_path_idx in [0..5] for paths [7,8,4,5,9,10]; choice1: 1=left, 2=right.
    """
    if choice1 == 1:
        return best_path_idx in LEFT_PATH_IDX
    if choice1 == 2:
        return best_path_idx in RIGHT_PATH_IDX
    return False


def correct2(value2, value3, choice1: int, choice2: int) -> bool:
    """
    Stage 2 correct if the chosen BRANCH attains the maximal subpath in that subtree.
      Left : compare max(3+7, 3+8) vs 4  -> choice2 in {1(node3), 2(node4)}
      Right: compare max(6+9, 6+10) vs 5 -> choice2 in {4(node6), 3(node5)}
    Note: choosing node3/node6 counts as correct iff that double-branch wins
          (specific leaf is judged at Stage 3).
    """
    sub = _subpaths_in_subtree(value2, value3, choice1)
    if sub.size != 3:
        return False

    if choice1 == 1:
        # indices in `sub`: 0->(3+7), 1->(3+8), 2->(4)
        best_is_double = (max(sub[0], sub[1]) >= sub[2])
        if best_is_double:
            return (choice2 == 1)  # node 3 branch
        else:
            return (choice2 == 2)  # node 4 single
    elif choice1 == 2:
        # indices in `sub`: 0->(6+9), 1->(6+10), 2->(5)
        best_is_double = (max(sub[0], sub[1]) >= sub[2])
        if best_is_double:
            return (choice2 == 4)  # node 6 branch
        else:
            return (choice2 == 3)  # node 5 single
    return False

def correct3(value3, choice1: int, choice2: int, choice3: int):
    """
    Stage 3 correct if, having picked the double-branch at Stage 2,
    the chosen leaf is the larger of the two.
      Left  : Stage2 must be node3; compare 7 vs 8; choice3 in {1=7, 2=8}
      Right : Stage2 must be node6; compare 9 vs 10; choice3 in {5=9, 6=10}
    If Stage 2 picked a single leaf branch, returns None (or False).
    """
    v3 = np.asarray(value3, dtype=float)
    if choice1 == 1 and choice2 == 1:
        best_leaf = 1 if v3[0] >= v3[1] else 2
        return (choice3 == best_leaf)
    if choice1 == 2 and choice2 == 4:
        best_leaf = 5 if v3[2] >= v3[3] else 6
        return (choice3 == best_leaf)
    return None  # no third stage

def subtree_vals(value2, value3, choice1, stage=2):
    """
    Utility accessor tailored to your analyses:
      stage=2 -> return the THREE subpath values in the chosen subtree
                 Left : [3+7, 3+8, 4]
                 Right: [6+9, 6+10, 5]
      stage=3 -> return the TWO leaf values if Stage 2 would be the double branch
                 Left : [7, 8]
                 Right: [9, 10]
    """
    v2 = np.asarray(value2, dtype=float)
    v3 = np.asarray(value3, dtype=float)
    if stage == 2:
        return _subpaths_in_subtree(v2, v3, choice1)
    elif stage == 3:
        if choice1 == 1:
            return v3[0:2]   # 7,8
        elif choice1 == 2:
            return v3[2:4]   # 9,10
        else:
            return np.array([])
    else:
        raise ValueError("stage must be 2 or 3")

def subtree_relation_code(path_rewards):
    """
    Code the structure of the TOP TWO full paths for Tree 7:
      First digit (1/2)  -> best path uses left/right subtree AND branch type
                           1x: left  (x=0 double, x=2 single[4])
                           2x: right (x=0 double, x=2 single[5])
      Second digit (0/1/2/3) -> where the SECOND-BEST path lies:
                           0: second in same subtree as best
                           1: second in the opposite subtree
                           2/3 kept for symmetry with single/double variants
    Returned set (examples):
      10, 11  -> best=left-double; second in left/right
      12, 13  -> best=left-single(4); second in left/right
      20, 21  -> best=right-double; second in right/left
      22, 23  -> best=right-single(5); second in right/left
    """
    p = np.asarray(path_rewards, dtype=float)
    order = np.argsort(-p)
    best, second = int(order[0]), int(order[1])

    # identify if best uses left-double/single or right-double/single
    # double branches correspond to paths through {7,8} (indices 0,1) or {9,10} (indices 4,5)
    if best in (0, 1):       # 7 or 8 -> left-double (via node 3)
        head = 10
    elif best == 2:          # 4 -> left-single
        head = 12
    elif best == 3:          # 5 -> right-single
        head = 22
    else:                    # 9 or 10 (indices 4,5) -> right-double (via node 6)
        head = 20

    same_subtree = int(_path_subtree(best) == _path_subtree(second))
    tail = 0 if same_subtree else 1
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
        # ---------------- basics ----------------
        trial['best_path_idx'] = int(np.argmax(trial['path_rewards']))

        # ---------------- difficulties ----------------
        # Stage 1: same definition as before
        diff1 = calculate_diff1(trial['path_rewards'])
        trial['diff1'] = round(diff1 * 2) / 2

        # Stage 2: new definition (three subpaths in chosen subtree; max - mean(others))
        diff2 = calculate_diff2(trial['value2'], trial['value3'], trial['choice1'])
        trial['diff2'] = round(diff2 * 2) / 2

        # Stage 3: if the branch chosen at stage 2 is the double-leaf branch; else -1.0
        diff3_raw = calculate_diff3(trial['value3'], trial['choice1'], trial['choice2']) \
                    if 'choice2' in trial else -1.0
        trial['diff3'] = (round(diff3_raw * 2) / 2) if diff3_raw != -1.0 else -1.0

        # Overall trial difficulty (kept consistent with your previous code)
        trial['difficulty'] = round(calculate_diff1(trial['path_rewards']) * 2) / 2

        # ---------------- correctness flags ----------------
        # Stage 1: subtree containing the global best path?
        trial['correct1'] = int(correct1(trial['best_path_idx'], trial['choice1']))

        # Stage 2: branch that yields the maximal subpath in that subtree?
        # (uses new signature: (value2, value3, choice1, choice2))
        trial['correct2'] = int(correct2(trial['value2'], trial['value3'],
                                         trial['choice1'], trial['choice2'])) \
                            if 'choice2' in trial else None

        # Stage 3: if double-branch was chosen, did the agent pick the better leaf?
        # correct3 returns True/False or None (when no third stage)
        if 'choice3' in trial and 'choice2' in trial and trial['choice3'] is not None:
            c3 = correct3(trial['value3'], trial['choice1'],
                          trial['choice2'], int(trial['choice3']))
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

        # ---------------- relation code ----------------
        trial['subtree_relation'] = subtree_relation_code(trial['path_rewards'])

        # ---------------- RT transforms ----------------
        # Exponentiate and cast to int if present
        for k in ('rt1', 'rt2', 'rt3', 'rt'):
            if k in trial and trial[k] is not None:
                trial[k] = int(np.exp(trial[k]))

    # keep trials with overall difficulty in [2, 10]
    trials_new = [t for t in trials if 2 <= t['difficulty'] <= 10]

    # Convert numpy scalar types to native Python before dumping
    trials_new = [convert_numpy_types(t) for t in trials_new]

    with open(output_file, 'w') as f:
        for t in trials_new:
            json.dump(t, f)
            f.write('\n')

    print(f"Added Information to {len(trials_new)} trials")
    print(f"Removed {len(trials) - len(trials_new)} trials")
    print(f"Saved to {output_file}")

def main():
    input_file = 'Tree7/data/trial7_v7.json'
    output_file = 'Tree7/data/Tree7_v3.json'
    add_info_to_json(input_file, output_file) 

if __name__ == "__main__":
    main() 