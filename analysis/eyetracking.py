# -*- coding: utf-8 -*-
"""
Fixation-by-stage-and-node aggregation with tree metadata.

What you get per (wid, trial_index, stage, node):
  - fixation_duration:   sum of dt while gaze is within AOI of this node
  - fixation_count:      number of contiguous segments assigned to this node
  - layer_type:          {'first_layer','second_layer','other'} inferred from graph
  - tree_type:           'tree1' (3 leaves) | 'tree2' (4 leaves) | 'other_<k>'
  - path:                spatial, left-to-right leaf numbering within subtree (kept from your current code)
  - node_identity:       graph-order identity: {L,R,LL,LR,RL,RR} depending on tree_type
  - choice:              1 if node was chosen in this trial (from behavior_obj['choice']), else 0
  - answer:              1 for best leaf (by reward) and its first-layer parent, else 0
"""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import pandas as pd


# ------------------------------- Utilities -------------------------------

def _normalize_trials(behavior_obj) -> List[dict]:
    if isinstance(behavior_obj, dict) and isinstance(behavior_obj.get("trials", None), list):
        return behavior_obj["trials"]
    if isinstance(behavior_obj, list):
        return behavior_obj
    raise ValueError("behavior_obj should be list[trial_dict] or {'trials': [...]}")

def _build_trial_index(trials_list: List[dict]) -> Dict[Tuple[Optional[str], int], dict]:
    idx = {}
    for t in trials_list:
        if not isinstance(t, dict) or "trial_index" not in t:
            continue
        tid = int(t["trial_index"])
        wid = t.get("wid", None)
        if wid is not None:
            idx[(str(wid), tid)] = t
        idx[(None, tid)] = t  # fallback
    return idx

def _get_start_idx(tinfo: dict, graph: List[List[int]], rewards: List[Optional[float]]) -> int:
    # 1) prefer first None in rewards
    start = None
    try:
        for i, rv in enumerate(rewards):
            if rv is None:
                start = i
                break
    except Exception:
        start = None
    # 2) explicit start
    if start is None:
        start = tinfo.get("start", None)
    # 3) fallback: max out-degree (tie -> smallest idx)
    if start is None:
        outdeg = [(i, len(graph[i]) if isinstance(graph[i], list) else 0) for i in range(len(graph))]
        if outdeg:
            max_d = max(d for _, d in outdeg)
            cand = [i for i, d in outdeg if d == max_d]
            start = min(cand) if cand else 0
        else:
            start = 0
    return int(start)

def _children(u: Optional[int], graph: List[List[int]], n_nodes: int) -> List[int]:
    if u is None or not (0 <= u < len(graph)):
        return []
    kids = graph[u]
    if not isinstance(kids, list):
        return []
    return [int(v) for v in kids if isinstance(v, int) and 0 <= v < n_nodes]

def _build_parent_map(graph: List[List[int]], n_nodes: int) -> Dict[int, Optional[int]]:
    parent = {i: None for i in range(n_nodes)}
    for u, kids in enumerate(graph):
        if not isinstance(kids, list):
            continue
        for v in kids:
            if isinstance(v, int) and 0 <= v < n_nodes:
                parent[v] = u
    return parent

def _infer_layers(graph: List[List[int]], rewards: List[Optional[float]], start: int, n_nodes: int) -> Tuple[Set[int], Set[int]]:
    first = set(_children(start, graph, n_nodes))
    second = set()
    for u in first:
        second.update(_children(u, graph, n_nodes))
    second.discard(start)
    second.difference_update(first)
    return first, second

def _tree_type(second_layer: Set[int]) -> str:
    k = len(second_layer)
    if k == 3:
        return "tree1"
    if k == 4:
        return "tree2"
    return f"other_{k}"

def _assign_node_identity_from_graph(graph: List[List[int]], rewards: List[Optional[float]], start: int, first: Set[int], second: Set[int], n_nodes: int) -> Tuple[Dict[int, Optional[str]], List[int], List[int]]:
    """
    Graph-based identity (no screen coordinates):
      - first-layer children of root (graph order): L, R
      - leaves under L: LL, LR (order by adjacency), under R: RL, RR
    Returns:
      node_identity map, left_leaves (2nd layer), right_leaves (2nd layer)
    """
    node_identity = {j: None for j in range(n_nodes)}

    first_list = _children(start, graph, n_nodes)  # ordered by adjacency
    left_child  = first_list[0] if len(first_list) >= 1 else None
    right_child = first_list[1] if len(first_list) >= 2 else None
    if left_child  is not None: node_identity[left_child]  = "L"
    if right_child is not None: node_identity[right_child] = "R"

    left_leaves  = [j for j in _children(left_child,  graph, n_nodes) if j in second]
    right_leaves = [j for j in _children(right_child, graph, n_nodes) if j in second]

    # assign leaf labels in adjacency order
    if len(left_leaves)  >= 1: node_identity[left_leaves[0]]  = "LL"
    if len(left_leaves)  >= 2: node_identity[left_leaves[1]]  = "LR"
    if len(right_leaves) >= 1: node_identity[right_leaves[0]] = "RL"
    if len(right_leaves) >= 2: node_identity[right_leaves[1]] = "RR"

    return node_identity, left_leaves, right_leaves

def _assign_path_spatial(centers: np.ndarray, graph: List[List[int]], start: int, first: Set[int], second: Set[int], n_nodes: int) -> Dict[int, Optional[str]]:
    """
    Spatial 'path' as in your current function:
      - determine left/right by x-order of first-layer children
      - then number leaves left->right within each side
      - propagate the leftmost leaf's path label to its first-layer parent
    """
    path_of = {j: None for j in range(n_nodes)}
    if not (isinstance(start, int) and 0 <= start < n_nodes):
        return path_of

    path_of[start] = "root"

    if not first:
        return path_of

    # use adjacency order (same as node_identity) instead of spatial order
    first_list = _children(start, graph, n_nodes)  # ordered by adjacency
    left_child  = first_list[0] if len(first_list) >= 1 else None
    right_child = first_list[1] if len(first_list) >= 2 else None

    # second-layer leaves per side
    def _kids(u):
        return [int(v) for v in (graph[u] if (u is not None and 0 <= u < len(graph) and isinstance(graph[u], list)) else [])
                if 0 <= v < n_nodes]

    left_leaves  = [j for j in _kids(left_child)  if j in second]
    right_leaves = [j for j in _kids(right_child) if j in second]

    # use adjacency order (same as node_identity) instead of spatial order
    # left_leaves and right_leaves are already in adjacency order from _kids()

    # number paths to match node_identity order: LL=path1, LR=path2, RL=path3, RR=path4
    if len(left_leaves) >= 1: path_of[left_leaves[0]] = "path1"   # LL
    if len(left_leaves) >= 2: path_of[left_leaves[1]] = "path2"   # LR
    if len(right_leaves) >= 1: path_of[right_leaves[0]] = "path3"  # RL
    if len(right_leaves) >= 2: path_of[right_leaves[1]] = "path4"  # RR

    # assign first-layer nodes as "left" and "right"
    if left_child is not None:
        path_of[left_child] = "left"
    if right_child is not None:
        path_of[right_child] = "right"

    return path_of

def _infer_choice_nodes(tinfo: dict, n_nodes: int) -> Set[int]:
    # Your example uses a list of node indices in key 'choice'
    val = tinfo.get("choice", [])
    if isinstance(val, (int, np.integer)):
        return {int(val)} if 0 <= int(val) < n_nodes else set()
    if isinstance(val, (list, tuple)):
        return {int(x) for x in val if isinstance(x, (int, np.integer)) and 0 <= int(x) < n_nodes}
    return set()

def _best_path_by_total_reward(rewards: List[Optional[float]], first_set: Set[int], second_set: Set[int], graph: List[List[int]], n_nodes: int) -> Set[int]:
    """
    Calculate the optimal path by finding the first_layer + second_layer pair with highest total reward.
    Returns the set containing both nodes in the optimal path.
    """
    def r_val(j):
        r = rewards[j]
        try:
            return float(r)
        except Exception:
            return float("-inf")
    
    best_path = None
    best_total_reward = float("-inf")
    
    # Iterate through all first layer nodes
    for first_node in first_set:
        first_reward = r_val(first_node)
        
        # Get all children (second layer nodes) of this first layer node
        children = _children(first_node, graph, n_nodes)
        second_layer_children = [child for child in children if child in second_set]
        
        # For each possible second layer node connected to this first layer node
        for second_node in second_layer_children:
            second_reward = r_val(second_node)
            total_reward = first_reward + second_reward
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_path = (first_node, second_node)
    
    # Return the nodes in the best path
    ans = set()
    if best_path is not None:
        ans.add(best_path[0])  # first layer node
        ans.add(best_path[1])  # second layer node
    
    return ans

def _best_leaf_and_parent(rewards: List[Optional[float]], left_leaves: List[int], right_leaves: List[int], parent_map: Dict[int, Optional[int]]) -> Set[int]:
    """
    [DEPRECATED] Choose best leaf by reward (ties broken by adjacency order: left list then right list).
    Mark that leaf and its first-layer parent.
    This function is kept for compatibility but should use _best_path_by_total_reward instead.
    """
    leaf_candidates = list(left_leaves) + list(right_leaves)

    def r_val(j):
        r = rewards[j]
        try:
            return float(r)
        except Exception:
            return float("-inf")

    best_leaf = None
    best_val = float("-inf")
    for j in leaf_candidates:
        v = r_val(j)
        if v > best_val:
            best_val = v
            best_leaf = j

    ans = set()
    if best_leaf is not None:
        ans.add(best_leaf)
        p = parent_map.get(best_leaf, None)
        if p is not None:
            ans.add(p)
    return ans


# ------------------------------- Main API -------------------------------

def compute_fixation_by_stage_and_node(eye_tracking_df: pd.DataFrame, behavior_obj, AOI_r: float = 60.0, include_zero_rows: bool = True) -> pd.DataFrame:
    """
    Compute per (wid, trial_index, stage(visit), node):
      - fixation_duration: sum of dt where AOI assigns to node
      - fixation_count   : number of contiguous runs assigned to node
      - layer_type       : {'first_layer','second_layer','other'} from graph
      - tree_type        : 'tree1'|'tree2'|'other_k' by # of second-layer nodes
      - path             : spatial path labels (kept from your current function)
      - node_identity    : graph-based identity labels {L,R,LL,LR,RL,RR}
      - choice           : 1 if node chosen by participant in this trial (stage-dependent), else 0
      - answer           : 1 if node belongs to optimal reward path (stage-dependent), else 0

    Stage-based choice/answer assignment:
      - Stage 0: choice=0, answer=0 for all nodes
      - Stage 1: choice=1/answer=1 only for first_layer nodes (if applicable)
      - Stage 2: choice=1/answer=1 only for second_layer nodes (if applicable)  
      - Stage 3+: choice=1/answer=1 for both first_layer and second_layer nodes (if applicable)
      
    Note: 'choice' reflects participant's actual decisions, 'answer' reflects optimal path
    based on total reward calculation (first_layer + second_layer reward sum)

    Returns columns:
      ['wid','trial_index','stage','node','layer_type','reward',
       'fixation_duration','fixation_count','tree_type','path','node_identity','choice','answer']
      (without 'wid' if the input gaze df has no 'wid' column)
    """
    trials_list = _normalize_trials(behavior_obj)
    trial_by_key = _build_trial_index(trials_list)

    df = eye_tracking_df.copy()
    df.columns = [c.strip() for c in df.columns]
    need = {"trial_index","Time","X","Y","visit"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"eye_tracking_df missing columns: {missing}")

    has_wid = "wid" in df.columns
    if has_wid:
        df["wid"] = df["wid"].astype(str)

    # coerce numeric types
    df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce").astype("Int64")
    df["Time"]        = pd.to_numeric(df["Time"], errors="coerce")
    df["X"]           = pd.to_numeric(df["X"], errors="coerce")
    df["Y"]           = pd.to_numeric(df["Y"], errors="coerce")
    df["visit"]       = pd.to_numeric(df["visit"], errors="coerce")

    df = df.dropna(subset=["trial_index","Time","X","Y","visit"]).copy()
    df["trial_index"] = df["trial_index"].astype(int)
    df["stage"] = df["visit"].round().astype(int)
    order_cols = ["wid","trial_index","Time"] if has_wid else ["trial_index","Time"]
    df = df.sort_values(order_cols).reset_index(drop=True)

    rows: List[dict] = []
    group_cols = ["wid","trial_index"] if has_wid else ["trial_index"]

    for key_vals, g in df.groupby(group_cols):
        if has_wid:
            wid_val, tid = key_vals
            tinfo = trial_by_key.get((wid_val, int(tid)), trial_by_key.get((None, int(tid))))
        else:
            tid = key_vals
            tinfo = trial_by_key.get((None, int(tid)))

        if tinfo is None:
            continue

        graph   = tinfo.get("graph", None)
        rewards = tinfo.get("rewards", None)
        centers = np.asarray(tinfo.get("node_positions", None), dtype=float)

        if graph is None or rewards is None or centers is None:
            raise ValueError(f"trial {tid}: behavior_obj must include node_positions / graph / rewards")

        n_nodes = len(rewards)

        # ---- root & layers ----
        start = _get_start_idx(tinfo, graph, rewards)
        first_set, second_set = _infer_layers(graph, rewards, start, n_nodes)

        # ---- meta labels ----
        tree_type = _tree_type(second_set)

        # graph-based identity (and adjacency-ordered leaves)
        node_identity_map, left_leaves, right_leaves = _assign_node_identity_from_graph(
            graph, rewards, start, first_set, second_set, n_nodes
        )

        # spatial path labels (keep your original behavior)
        path_map = _assign_path_spatial(
            centers, graph, start, first_set, second_set, n_nodes
        )

        # choice & answer
        choice_nodes = _infer_choice_nodes(tinfo, n_nodes)
        choice_nodes.discard(start)            # <-- ensure start node is never marked as choice

        parent_map   = _build_parent_map(graph, n_nodes)
        answer_nodes = _best_path_by_total_reward(rewards, first_set, second_set, graph, n_nodes)

        # ---- per-trial dt ----
        tvals = g["Time"].to_numpy()
        dt = np.zeros_like(tvals, dtype=float)
        if len(tvals) > 1:
            dt[:-1] = np.maximum(0.0, tvals[1:] - tvals[:-1])
        g = g.assign(dt=dt)

        # ---- AOI assignment (nearest center within AOI_r) ----
        pts = g[["X","Y"]].to_numpy(float)
        dx = pts[:, None, 0] - centers[None, :, 0]
        dy = pts[:, None, 1] - centers[None, :, 1]
        dist2 = dx * dx + dy * dy
        nearest = np.argmin(dist2, axis=1)
        min_dist = np.sqrt(np.min(dist2, axis=1))
        node_idx = np.where(min_dist <= AOI_r, nearest, -1)
        g = g.assign(node_idx=node_idx)

        # ---- aggregate per stage ----
        for stage_val, sg in g.groupby("stage"):
            dur_by_node = (
                sg[sg["node_idx"] >= 0]
                .groupby("node_idx", observed=True)["dt"]
                .sum()
            )
            # contiguous segments per node
            sg2 = sg.copy()
            seg_id = (sg2["node_idx"] != sg2["node_idx"].shift(1)).cumsum().rename("segment_id")
            tmp = sg2.assign(segment_id=seg_id)
            count_by_node = (
                tmp[tmp["node_idx"] >= 0]
                .groupby(["node_idx","segment_id"], observed=True)
                .size().reset_index(name="n")
                .groupby("node_idx", observed=True)["segment_id"]
                .nunique()
            )

            if include_zero_rows:
                all_idx = pd.Index(range(n_nodes), name="node_idx")
                dur_by_node   = dur_by_node.reindex(all_idx, fill_value=0.0)
                count_by_node = count_by_node.reindex(all_idx, fill_value=0)

            node_iter = range(n_nodes) if include_zero_rows else \
                sorted(set(map(int, dur_by_node.index.tolist())) | set(map(int, count_by_node.index.tolist())))

            for j in node_iter:
                # Stage-specific choice and answer assignment
                is_first_layer = j in first_set
                is_second_layer = j in second_set
                stage = int(stage_val)
                
                # Default to 0
                choice_val = 0
                answer_val = 0
                
                # Stage-based logic for choice and answer
                if stage == 0:
                    # Stage 0: all nodes have choice=0, answer=0
                    choice_val = 0
                    answer_val = 0
                elif stage == 1:
                    # Stage 1: only first_layer nodes can have choice=1 or answer=1
                    if is_first_layer:
                        choice_val = 1 if j in choice_nodes else 0
                        answer_val = 1 if j in answer_nodes else 0
                    else:
                        choice_val = 0
                        answer_val = 0
                elif stage == 2:
                    # Stage 2: only second_layer nodes can have choice=1 or answer=1
                    if is_second_layer:
                        choice_val = 1 if j in choice_nodes else 0
                        answer_val = 1 if j in answer_nodes else 0
                    else:
                        choice_val = 0
                        answer_val = 0
                elif stage >= 3:
                    # Stage 3+: both first_layer and second_layer nodes can have choice=1 or answer=1
                    if is_first_layer or is_second_layer:
                        choice_val = 1 if j in choice_nodes else 0
                        answer_val = 1 if j in answer_nodes else 0
                    else:
                        choice_val = 0
                        answer_val = 0
                
                row = {
                    "trial_index": int(tid),
                    "stage": stage,
                    "node": int(j),
                    "layer_type": ("first_layer" if is_first_layer else
                                   "second_layer" if is_second_layer else "other"),
                    "reward": rewards[j] if 0 <= j < len(rewards) else None,
                    "fixation_duration": float(dur_by_node.get(j, 0.0)),
                    "fixation_count": int(count_by_node.get(j, 0)),
                    "tree_type": tree_type,
                    "path": path_map.get(j, None),
                    "node_identity": node_identity_map.get(j, None),
                    "choice": choice_val,
                    "answer": answer_val,
                }
                if has_wid:
                    row["wid"] = str(wid_val)
                rows.append(row)

    result = pd.DataFrame(rows)
    base_cols = ["trial_index","stage","node","layer_type","reward",
                 "fixation_duration","fixation_count","tree_type","path","node_identity","choice","answer"]
    if has_wid:
        cols = ["wid"] + base_cols
    else:
        cols = base_cols
    result = result[cols].sort_values(cols[:4]).reset_index(drop=True)
    return result