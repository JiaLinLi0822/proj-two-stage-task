import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_fixation_bars(res: pd.DataFrame):
    """
    Bar plots for tree1 and tree2:
      - X: Stage 1, Stage 2
      - Color (bars): layer_type ∈ {first_layer, second_layer, other}
      - Y: fixation_duration (participant mean across trials)
      - Error bars: SEM across participants
      - Dots: each participant (wid) as a jittered scatter on each bar
    """
    # ---- checks ----
    needed = {'tree_type','stage','layer_type','fixation_duration','trial_index','wid'}
    missing = needed - set(res.columns)
    if missing:
        raise ValueError(f"`res` missing columns: {missing}")

    # keep only stage 1/2
    df = res[res['stage'].isin([1, 2])].copy()

    # 1) Sum fixation across nodes within each (wid, trial, tree_type, stage, layer_type)
    df_trial = (
        df.groupby(['wid','trial_index','tree_type','stage','layer_type'], observed=True)['fixation_duration']
          .sum()
          .reset_index(name='fix_dur_sum')
    )

    # 2) Mean across trials per participant (wid)
    df_wid = (
        df_trial.groupby(['wid','tree_type','stage','layer_type'], observed=True)['fix_dur_sum']
               .mean()
               .reset_index(name='fix_dur_mean')
    )

    # 3) Group means & SEM across participants
    def _sem(x: pd.Series) -> float:
        n = x.count()
        if n <= 1:
            return 0.0
        return float(x.std(ddof=1) / math.sqrt(n))

    group_stats = (
        df_wid.groupby(['tree_type','stage','layer_type'], observed=True)['fix_dur_mean']
              .agg(mean='mean', sem=_sem, n='count')
              .reset_index()
    )

    # ---- plotting ----
    layer_order = ['first_layer', 'second_layer', 'other']
    stage_order = [1, 2]
    bar_width = 0.25
    np.random.seed(0)  # for reproducible jitter
    
    # Define colors for each layer type
    layer_colors = {
        'first_layer': '#2E8B57',   # Sea green
        'second_layer': '#4169E1',  # Royal blue
        'other': '#DC143C'          # Crimson
    }

    def _plot_one_tree(tree_type: str):
        sub_stats = group_stats[group_stats['tree_type'] == tree_type].copy()
        sub_wid   = df_wid[df_wid['tree_type'] == tree_type].copy()
        if sub_stats.empty:
            return

        fig = plt.figure(figsize=(3.35, 2.5))
        ax = plt.gca()

        x_base = np.arange(len(stage_order))  # 0 -> Stage 1, 1 -> Stage 2
        
        # Store participant positions for connecting lines
        participant_positions = {}

        for li, layer in enumerate(layer_order):
            offsets = (li - 1) * bar_width  # -0.25, 0, +0.25
            xpos = x_base + offsets
            layer_color = layer_colors[layer]

            # bar heights and SEMs for each stage
            means, sems = [], []
            for st in stage_order:
                row = sub_stats[(sub_stats['stage'] == st) & (sub_stats['layer_type'] == layer)]
                if len(row) == 1:
                    means.append(float(row['mean'].iloc[0]))
                    sems.append(float(row['sem'].iloc[0]))
                else:
                    means.append(0.0)
                    sems.append(0.0)

            # draw bars + error bars with consistent colors
            ax.bar(xpos, means, bar_width, yerr=sems, capsize=4, 
                  label=layer.replace('_',' '), color=layer_color, alpha=0.7)

            # scatter per participant (jittered) with consistent colors and reduced transparency
            for i, st in enumerate(stage_order):
                wid_vals = sub_wid[(sub_wid['stage'] == st) & (sub_wid['layer_type'] == layer)]
                if wid_vals.empty:
                    continue
                x_center = xpos[i]
                
                # Generate consistent jitter for each participant
                for idx, (_, row) in enumerate(wid_vals.iterrows()):
                    wid = row['wid']
                    fix_dur = row['fix_dur_mean']
                    
                    # Use participant ID for consistent jitter
                    np.random.seed(hash(f"{wid}_{layer}") % 2**32)
                    x_jitter = x_center + (np.random.rand() - 0.5) * (bar_width * 0.6)
                    
                    # Store position for connecting lines
                    if wid not in participant_positions:
                        participant_positions[wid] = {}
                    if layer not in participant_positions[wid]:
                        participant_positions[wid][layer] = {}
                    participant_positions[wid][layer][st] = (x_jitter, fix_dur)
                    
                    # Plot scatter point with reduced transparency
                    ax.scatter(x_jitter, fix_dur, s=18, color=layer_color, alpha=0.6)

        # # Connect same participant across stages for each layer type
        # for wid, layers_data in participant_positions.items():
        #     for layer, stages_data in layers_data.items():
        #         if len(stages_data) == 2:  # has both stage 1 and stage 2
        #             x_coords = [stages_data[st][0] for st in stage_order if st in stages_data]
        #             y_coords = [stages_data[st][1] for st in stage_order if st in stages_data]
                    
        #             if len(x_coords) == 2:  # ensure we have both stages
        #                 ax.plot(x_coords, y_coords, color=layer_colors[layer], 
        #                        alpha=0.3, linewidth=0.8, zorder=1)
        
        # Connect different layer types within each stage for the same participant (black lines)
        for wid, layers_data in participant_positions.items():
            for st in stage_order:
                # Get positions for all layer types in this stage for this participant
                stage_positions = []
                for layer in layer_order:
                    if layer in layers_data and st in layers_data[layer]:
                        stage_positions.append(layers_data[layer][st])
                
                # Connect all points within this stage if we have multiple layer types
                if len(stage_positions) >= 2:
                    x_coords = [pos[0] for pos in stage_positions]
                    y_coords = [pos[1] for pos in stage_positions]
                    
                    # Draw lines connecting all layer types within this stage
                    for i in range(len(stage_positions) - 1):
                        ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], 
                               color='black', alpha=0.4, linewidth=0.6, zorder=0)

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"Stage {s}" for s in stage_order])
        ax.set_ylabel("Fixation duration")
        # ax.set_title(f"{tree_type} (group means ± SEM; dots = participants)")
        ax.legend(title="Layer type", fontsize=7, loc='upper right', title_fontsize=7)
        ax.margins(x=0.05)
        plt.tight_layout()
        plt.show()

    # make figures (only if present)
    for tt in ['tree1', 'tree2']:
        _plot_one_tree(tt)
