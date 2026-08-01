#!/usr/bin/env python3

import json
import os


INPUT_FILE = "Tree2/data/pda/model1.json"
OUTPUT_FILE = "Tree2/data/pda/model1_cleaned.json"

RT1_MIN_MS = 500
RT1_MAX_MS = 15000
RT2_MIN_MS = 300
RT2_MAX_MS = 10000


def main():
    with open(INPUT_FILE, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    n_before = len(records)
    kept = []
    n_rt1_below_min = n_rt1_above_max = n_rt2_below_min = n_rt2_above_max = 0
    for r in records:
        rt1 = r.get("rt1")
        rt2 = r.get("rt2")
        if rt1 is None or rt2 is None:
            continue
        try:
            rt1 = float(rt1)
            rt2 = float(rt2)
        except (TypeError, ValueError):
            continue
        if RT1_MIN_MS <= rt1 <= RT1_MAX_MS and RT2_MIN_MS <= rt2 <= RT2_MAX_MS:
            kept.append(r)
        else:
            if rt1 < RT1_MIN_MS:
                n_rt1_below_min += 1
            if rt1 > RT1_MAX_MS:
                n_rt1_above_max += 1
            if rt2 < RT2_MIN_MS:
                n_rt2_below_min += 1
            if rt2 > RT2_MAX_MS:
                n_rt2_above_max += 1

    n_after = len(kept)
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    n_dropped = n_before - n_after
    print(f"Input:  {INPUT_FILE}  ({n_before} trials)")
    print(f"Output: {OUTPUT_FILE}  ({n_after} trials, dropped {n_dropped})")
    print(f"RT filter: rt1 [{RT1_MIN_MS}, {RT1_MAX_MS}] ms, rt2 [{RT2_MIN_MS}, {RT2_MAX_MS}] ms")
    if n_dropped > 0:
        print(f"Dropped due to RT (a trial may have multiple): rt1 < min: {n_rt1_below_min}, rt1 > max: {n_rt1_above_max}, rt2 < min: {n_rt2_below_min}, rt2 > max: {n_rt2_above_max}")

if __name__ == "__main__":
    main()
