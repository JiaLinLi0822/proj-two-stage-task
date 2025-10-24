import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_fixation_by_stage_and_layer(df, save_path=None):
    """
    Plot fixation duration by stage and layer type for tree1 and tree2.
    
    Parameters:
    - df: DataFrame with columns ['wid', 'trial_index', 'stage', 'layer_type', 'tree_type', 'fixation_duration']
    - save_path: Optional path to save the figure
    """
    
    # Filter for tree1 and tree2 only
    df_filtered = df[df['tree_type'].isin(['tree1', 'tree2'])].copy()
    
    # Aggregate by participant, stage, layer_type, and tree_type
    agg_data = df_filtered.groupby(['wid', 'stage', 'layer_type', 'tree_type'])['fixation_duration'].sum().reset_index()
    
    # Create the figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors for different layer types
    colors = {
        'first_layer': '#2E8B57',   # Sea green
        'second_layer': '#4169E1',  # Royal blue
        'other': '#DC143C'          # Crimson
    }
    
    # Define stage labels
    stage_labels = ['Stage 1', 'Stage 2']
    
    for i, tree_type in enumerate(['tree1', 'tree2']):
        ax = axes[i]
        
        # Filter data for current tree type
        tree_data = agg_data[agg_data['tree_type'] == tree_type]
        
        # Get unique stages and layer types
        stages = sorted(tree_data['stage'].unique())
        layer_types = ['first_layer', 'second_layer', 'other']
        
        # Calculate group-level statistics
        group_stats = tree_data.groupby(['stage', 'layer_type'])['fixation_duration'].agg(['mean', 'sem']).reset_index()
        
        # Set up bar positions
        x = np.arange(len(stages))
        width = 0.25
        
        # Plot bars for each layer type
        for j, layer_type in enumerate(layer_types):
            layer_stats = group_stats[group_stats['layer_type'] == layer_type]
            
            if len(layer_stats) > 0:
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
                
                # Plot bars with error bars
                bars = ax.bar(x + j * width, means, width, 
                             label=layer_type.replace('_', ' ').title(),
                             color=colors[layer_type], alpha=0.7, 
                             yerr=sems, capsize=3)
                
                # Add individual participant data as scatter points
                for k, stage in enumerate(stages):
                    stage_layer_data = tree_data[(tree_data['stage'] == stage) & 
                                                (tree_data['layer_type'] == layer_type)]
                    
                    if len(stage_layer_data) > 0:
                        # Add some jitter to x positions
                        x_pos = x[k] + j * width
                        jitter = np.random.normal(0, width/8, len(stage_layer_data))
                        
                        ax.scatter(x_pos + jitter, stage_layer_data['fixation_duration'], 
                                 color=colors[layer_type], alpha=0.6, s=15, edgecolors='white', linewidth=0.5)
        
        # Customize the subplot
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Fixation Duration (ms)', fontsize=12)
        ax.set_title(f'{tree_type.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(stage_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title
    fig.suptitle('Fixation Duration by Stage and Layer Type', fontsize=16, fontweight='bold', y=1.02)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("="*50)
    
    for tree_type in ['tree1', 'tree2']:
        print(f"\n{tree_type.upper()}:")
        tree_data = agg_data[agg_data['tree_type'] == tree_type]
        
        summary = tree_data.groupby(['stage', 'layer_type'])['fixation_duration'].agg([
            'count', 'mean', 'std', 'sem'
        ]).round(2)
        
        print(summary)

# Example usage:
if __name__ == "__main__":
    # Load your data (replace with actual data loading)
    # df = pd.read_csv('your_fixation_data.csv')
    
    # For demonstration, create sample data
    np.random.seed(42)
    
    wids = [f'P{i}' for i in range(1, 21)]  # 20 participants
    stages = [1, 2]
    layer_types = ['first_layer', 'second_layer', 'other']
    tree_types = ['tree1', 'tree2']
    
    sample_data = []
    
    for wid in wids:
        for tree_type in tree_types:
            for stage in stages:
                for layer_type in layer_types:
                    # Generate realistic fixation durations
                    base_duration = 200 + np.random.normal(0, 50)
                    if layer_type == 'first_layer':
                        base_duration *= 1.2
                    elif layer_type == 'second_layer':
                        base_duration *= 1.5
                    
                    if stage == 2:
                        base_duration *= 1.3
                    
                    duration = max(0, base_duration + np.random.normal(0, 30))
                    
                    sample_data.append({
                        'wid': wid,
                        'stage': stage,
                        'layer_type': layer_type,
                        'tree_type': tree_type,
                        'fixation_duration': duration,
                        'trial_index': 1
                    })
    
    df_sample = pd.DataFrame(sample_data)
    
    # Plot the figure
    plot_fixation_by_stage_and_layer(df_sample, save_path='fixation_by_stage_plot.png')
