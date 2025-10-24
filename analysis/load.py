import os
import json
import pandas as pd
from pandas import json_normalize

def _load_behavior_trials(obj, wid):
    """
    Normalize a behavior object (list / {'trials': [...]} / single-trial dict)
    into a list of trial dicts. Adds 'wid' to each trial.
    """
    trials = []
    if isinstance(obj, dict) and "trials" in obj and isinstance(obj["trials"], list):
        trials = obj["trials"]
    elif isinstance(obj, list):
        trials = obj
    elif isinstance(obj, dict) and "trial_index" in obj:
        trials = [obj]
    else:
        # Unknown format; return empty
        return []

    out = []
    for t in trials:
        if isinstance(t, dict):
            td = dict(t)  # shallow copy
            td["wid"] = wid
            # ensure trial_index is int if present
            if "trial_index" in td:
                try:
                    td["trial_index"] = int(td["trial_index"])
                except Exception:
                    pass
            out.append(td)
    return out


def load_all_participants(csv_path, json_path, participants):
    eye_parts = []
    trials_all = []

    for wid in participants:
        # ---- eyetracking ----
        csv_file = os.path.join(csv_path, f"{wid}.csv")
        if os.path.isfile(csv_file):
            try:
                df_eye = pd.read_csv(csv_file)
                # add subject column
                df_eye["wid"] = wid
                eye_parts.append(df_eye)
            except Exception as e:
                print(f"[WARN] Failed to read CSV for {wid}: {e}")
        else:
            print(f"[INFO] CSV missing for {wid}: {csv_file}")

        # ---- behavioral ----
        json_file = os.path.join(json_path, f"{wid}.json")
        if os.path.isfile(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                trials = _load_behavior_trials(obj, wid)
                trials_all.extend(trials)
            except Exception as e:
                print(f"[WARN] Failed to read JSON for {wid}: {e}")
        else:
            print(f"[INFO] JSON missing for {wid}: {json_file}")

    # concat eyetracking across subjects (align columns by name)
    all_eye = pd.concat(eye_parts, ignore_index=True) if eye_parts else pd.DataFrame()

    # a flat dataframe view of the trials (useful for quick filtering/groupby)
    all_trials_df = json_normalize(trials_all) if trials_all else pd.DataFrame()

    return all_eye, trials_all, all_trials_df