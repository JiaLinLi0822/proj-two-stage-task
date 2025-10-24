#!/usr/bin/env python3
"""
Quick plotting script - assumes you already have the processed fixation data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quick_plot_fixation(df):
    """
    Quick plot function for fixation data.
    Assumes df has columns: ['wid', 'stage', 'layer_type', 'tree_type', 'fixation_duration']
    """
    
    # Filter for stages 1 and 2, tree1 and tree2
    df_plot = df[
        (df['stage'].isin([1, 2])) & 
        (df['tree_type'].isin(['tree1', 'tree2']))
    ].copy()
    
    # Aggregate by participant
    agg_data = df_plot.groupby(['wid', 'stage', 'layer_type', 'tree_type'])['fixation_duration'].sum().reset_index()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for each layer type
    colors = {'other': '#DC143C', 'first_layer': '#2E8B57', 'second_layer': '#4169E1'}
    labels = {'other': 'Root Node', 'first_layer': 'First Layer', 'second_layer': 'Second Layer'}
    
    for ax, tree in zip([ax1, ax2], ['tree1', 'tree2']):
        tree_data = agg_data[agg_data['tree_type'] == tree]
        
        # Calculate means and SEMs
        stats = tree_data.groupby(['stage', 'layer_type'])['fixation_duration'].agg(['mean', 'sem']).reset_index()
        
        # Bar plot setup
        stages = [1, 2]
        x = np.arange(len(stages))
        width = 0.25
        
        for i, layer in enumerate(['other', 'first_layer', 'second_layer']):
            layer_stats = stats[stats['layer_type'] == layer]
            
            means = []
            sems = []
            for stage in stages:
                stage_data = layer_stats[layer_stats['stage'] == stage]
                if len(stage_data) > 0:
                    means.append(stage_data['mean'].iloc[0])
                    sems.append(stage_data['sem'].iloc[0])
                else:
                    means.append(0)
                    sems.append(0)
            
            # Plot bars
            ax.bar(x + i*width, means, width, label=labels[layer], 
                  color=colors[layer], alpha=0.7, yerr=sems, capsize=3)
            
            # Add scatter points for individual participants
            for j, stage in enumerate(stages):
                participant_data = tree_data[
                    (tree_data['stage'] == stage) & 
                    (tree_data['layer_type'] == layer)
                ]
                if len(participant_data) > 0:
                    jitter = np.random.normal(0, width/10, len(participant_data))
                    ax.scatter(x[j] + i*width + jitter, participant_data['fixation_duration'],
                             color=colors[layer], alpha=0.5, s=15, edgecolors='white', linewidth=0.5)
        
        # Styling
        ax.set_title(f'{tree.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Fixation Duration (ms)', fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Stage 1', 'Stage 2'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.suptitle('Fixation Duration by Stage and Layer Type', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('fixation_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# If you want to test with sample data:
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    data = []
    
    for wid in [f'P{i}' for i in range(1, 16)]:  # 15 participants
        for tree in ['tree1', 'tree2']:
            for stage in [1, 2]:
                for layer in ['other', 'first_layer', 'second_layer']:
                    # Generate realistic durations
                    base = 200
                    if layer == 'first_layer': base = 300
                    elif layer == 'second_layer': base = 400
                    if stage == 2: base *= 1.2
                    
                    duration = max(0, np.random.normal(base, 50))
                    data.append({
                        'wid': wid, 'stage': stage, 'layer_type': layer, 
                        'tree_type': tree, 'fixation_duration': duration
                    })
    
    df_sample = pd.DataFrame(data)
    quick_plot_fixation(df_sample)
    
    print("Sample plot created. Replace df_sample with your actual data.")
