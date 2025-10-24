#!/usr/bin/env python3
"""
Plot fixation duration results by stage and layer type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.eyetracking import compute_fixation_by_stage_and_node
import json

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_and_process_data():
    """Load and process the eyetracking and behavioral data."""
    
    # Load eyetracking data (adjust path as needed)
    try:
        eye_df = pd.read_json('/Users/lijialin/Desktop/Research/proj-two-stage-task/data/processed/trial_eyetracking.json')
    except:
        print("Could not load trial_eyetracking.json, please adjust the path")
        return None
    
    # Load behavioral data (adjust path as needed)
    try:
        with open('/Users/lijialin/Desktop/Research/proj-two-stage-task/data/Tree1_v3.json', 'r') as f:
            behavior_tree1 = json.load(f)
        with open('/Users/lijialin/Desktop/Research/proj-two-stage-task/data/Tree2_v3.json', 'r') as f:
            behavior_tree2 = json.load(f)
    except:
        print("Could not load behavioral data files, please adjust the paths")
        return None
    
    # Process the data
    results_tree1 = compute_fixation_by_stage_and_node(eye_df, behavior_tree1)
    results_tree2 = compute_fixation_by_stage_and_node(eye_df, behavior_tree2)
    
    # Combine results
    results_tree1['tree_type'] = 'tree1'
    results_tree2['tree_type'] = 'tree2'
    combined_df = pd.concat([results_tree1, results_tree2], ignore_index=True)
    
    return combined_df

def plot_fixation_by_stage_and_layer(df):
    """Create bar plots for fixation duration by stage and layer type."""
    
    # Filter for tree1 and tree2, and stages 1 and 2
    df_filtered = df[
        (df['tree_type'].isin(['tree1', 'tree2'])) & 
        (df['stage'].isin([1, 2]))
    ].copy()
    
    # Aggregate by participant, stage, layer_type, and tree_type
    agg_data = df_filtered.groupby(['wid', 'stage', 'layer_type', 'tree_type'])['fixation_duration'].sum().reset_index()
    
    # Create the figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors for different layer types
    colors = {
        'first_layer': '#2E8B57',   # Sea green
        'second_layer': '#4169E1',  # Royal blue  
        'other': '#DC143C'          # Crimson (for root node)
    }
    
    for i, tree_type in enumerate(['tree1', 'tree2']):
        ax = axes[i]
        
        # Filter data for current tree type
        tree_data = agg_data[agg_data['tree_type'] == tree_type]
        
        # Get stages and layer types
        stages = [1, 2]
        layer_types = ['other', 'first_layer', 'second_layer']  # 'other' is root node
        layer_labels = ['Root Node', 'First Layer', 'Second Layer']
        
        # Calculate group-level statistics
        group_stats = tree_data.groupby(['stage', 'layer_type'])['fixation_duration'].agg(['mean', 'sem']).reset_index()
        
        # Set up bar positions
        x = np.arange(len(stages))
        width = 0.25
        
        # Plot bars for each layer type
        for j, (layer_type, label) in enumerate(zip(layer_types, layer_labels)):
            means = []
            sems = []
            
            for stage in stages:
                stage_data = group_stats[(group_stats['stage'] == stage) & 
                                       (group_stats['layer_type'] == layer_type)]
                if len(stage_data) > 0:
                    means.append(stage_data['mean'].iloc[0])
                    sems.append(stage_data['sem'].iloc[0])
                else:
                    means.append(0)
                    sems.append(0)
            
            # Plot bars with error bars
            ax.bar(x + j * width, means, width, 
                  label=label, color=colors[layer_type], alpha=0.7, 
                  yerr=sems, capsize=3)
            
            # Add individual participant data as scatter points
            for k, stage in enumerate(stages):
                stage_layer_data = tree_data[(tree_data['stage'] == stage) & 
                                           (tree_data['layer_type'] == layer_type)]
                
                if len(stage_layer_data) > 0:
                    # Add jitter to x positions
                    x_pos = x[k] + j * width
                    n_points = len(stage_layer_data)
                    jitter = np.random.normal(0, width/10, n_points)
                    
                    ax.scatter(x_pos + jitter, stage_layer_data['fixation_duration'], 
                             color=colors[layer_type], alpha=0.5, s=20, 
                             edgecolors='white', linewidth=0.5)
        
        # Customize subplot
        ax.set_xlabel('Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fixation Duration (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'{tree_type.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Stage 1', 'Stage 2'])
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Adjust layout and add title
    plt.tight_layout()
    fig.suptitle('Fixation Duration by Stage and Layer Type', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Save the figure
    plt.savefig('/Users/lijialin/Desktop/Research/proj-two-stage-task/fixation_by_stage_plot.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for tree_type in ['tree1', 'tree2']:
        print(f"\n{tree_type.upper()}:")
        print("-" * 30)
        
        tree_data = agg_data[agg_data['tree_type'] == tree_type]
        
        summary = tree_data.groupby(['stage', 'layer_type'])['fixation_duration'].agg([
            'count', 'mean', 'std', 'sem'
        ]).round(2)
        
        print(summary)
        print()

if __name__ == "__main__":
    # Load and process the data
    print("Loading and processing eyetracking data...")
    df = load_and_process_data()
    
    if df is not None:
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Tree types: {df['tree_type'].unique()}")
        print(f"Stages: {sorted(df['stage'].unique())}")
        print(f"Layer types: {df['layer_type'].unique()}")
        print(f"Participants: {len(df['wid'].unique())}")
        
        # Create the plot
        print("\nCreating plots...")
        plot_fixation_by_stage_and_layer(df)
        
        print("Plot saved as 'fixation_by_stage_plot.png'")
    else:
        print("Failed to load data. Please check file paths.")
