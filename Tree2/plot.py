import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

model_number = 6
# Load data
with open("Tree2/data/Tree2.json", "r") as f_real:
    real_data = [json.loads(line) for line in f_real]
for entry in real_data:
    entry["source"] = "Human"
    entry["path"] = entry["rewards"]

# with open("Tree2/data/analytical/model6_analytical.json", "r") as f_sim:
#     model6_analytical_data = [json.loads(line) for line in f_sim]
# for entry in model6_analytical_data:
#     entry["source"] = "Model6_Analytical"

with open("Tree2/data/analytical/model6.json", "r") as f_sim:
    model6_analytical_data = [json.loads(line) for line in f_sim]
for entry in model6_analytical_data:
    entry["source"] = "Analytical"

with open("Tree2/data/rss/model1.json", "r") as f_sim:
    model1_RSS_data = [json.loads(line) for line in f_sim]
for entry in model1_RSS_data:
    entry["source"] = "TS_RSS"

with open("Tree2/data/rss/model6_RSS.json", "r") as f_sim:
    model6_RSS_data = [json.loads(line) for line in f_sim]
for entry in model6_RSS_data:
    entry["source"] = "FG_RSS"

# with open("Tree2/data/pda/model6_1e-8.json", "r") as f_sim:
#     model6_1e_8 = [json.loads(line) for line in f_sim]
# for entry in model6_1e_8:
#     entry["source"] = "PDA_1e-8"

with open("Tree2/data/pda/model6_1e-16.json", "r") as f_sim:
    model6_PDA_data = [json.loads(line) for line in f_sim]
for entry in model6_PDA_data:
    entry["source"] = "FG_PDA"  

with open("Tree2/data/pda/model1_cleaned.json", "r") as f_sim:
    model1_PDA_data = [json.loads(line) for line in f_sim]
for entry in model1_PDA_data:
    entry["source"] = "TS_PDA"

# with open("Tree2/data/ibs/model1.json", "r") as f_sim:
#     model1_IBS_data = [json.loads(line) for line in f_sim]
# for entry in model1_IBS_data:
#     entry["source"] = "Model1_IBS"

# with open("Tree2/data/pda/model6_pda2.json", "r") as f_sim:
#     model6_pda2_data = [json.loads(line) for line in f_sim]
# for entry in model6_pda2_data:
#     entry["source"] = "model6_pda_1e-16"

# with open("Tree2/data/pda/model6_pda3.json", "r") as f_sim:
#     model6_pda3_data = [json.loads(line) for line in f_sim]
# for entry in model6_pda3_data:
#     entry["source"] = "model6_pda3"

# with open("Tree2/data/pda/model15_pda.json", "r") as f_sim:
#     model15_pda_data = [json.loads(line) for line in f_sim]
# for entry in model15_pda_data:
#     entry["source"] = "model15_pda"

# with open("Tree2/data/rss/model6_RSS.json", "r") as f_sim:
#     model6_RSS_data = [json.loads(line) for line in f_sim]
# for entry in model6_RSS_data:
#     entry["source"] = "model6_RSS"

# with open("Tree2/data/ibs/model6_ibs.json", "r") as f_sim:
#     model6_ibs_data = [json.loads(line) for line in f_sim]
# for entry in model6_ibs_data:
#     entry["source"] = "model6_ibs"

# with open("Tree2/data/analytical/model6_analytical.json", "r") as f_sim:
#     model6_analytical_data = [json.loads(line) for line in f_sim]
# for entry in model6_analytical_data:
#     entry["source"] = "model6_analytical"

    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 9,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.linewidth': 0.75,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.loc': 'upper right'
    })

df = pd.DataFrame(real_data + model1_RSS_data + model1_PDA_data + model6_RSS_data + model6_PDA_data)

# Helper function
def subtree_vals(path_value, choice1):
    if choice1 == 1:
        return path_value[0:2]
    else:
        return path_value[2:4]

def subtree_relation_code(rewards):
    idx_desc = np.argsort(rewards)[::-1]
    best, second, third, worst = idx_desc
    subtree = lambda i: 0 if i < 2 else 1
    
    if subtree(best) == subtree(second):
        return 1
    elif subtree(best) == subtree(third):
        return 2
    elif subtree(best) == subtree(worst):
        return 3
    else:
        return np.nan

# Compute derived columns
df['best_path_idx'] = df['path'].apply(lambda v: int(np.argmax(v)))
df['correct1'] = df.apply(
    lambda r: (r['best_path_idx'] < 2 and r['choice1'] == 1) or 
              (r['best_path_idx'] >= 2 and r['choice1'] == 2),
    axis=1
)
df['correct2'] = df.apply(
    lambda r: r['value2'][r['choice2'] - 1] == max(subtree_vals(r['value2'], r['choice1'])),
    axis=1
)
df['correct_all'] = df['correct1'] & df['correct2']
df['subtree_relation'] = df['path'].apply(subtree_relation_code)

# Set font sizes (must be set before creating subplots)
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
})

# Source colors: Human black, model1/model6 RSS/PDA as specified
# SOURCE_COLORS = {
#     'Human': 'black',
#     'TS_RSS': '#335372',
#     'TS_PDA': '#4F7BA8',
#     'FG_RSS': '#E25659',
#     'FG_PDA': '#EB8386',
# }

SOURCE_COLORS = {
    'Human': '#1072BD',
    'TS_RSS': '#77AE43',
    'TS_PDA': '#EDB021',
    'FG_RSS': '#D7592C',
    'FG_PDA': '#7F318D',
}

def get_source_color(source):
    return SOURCE_COLORS.get(source, 'gray')

unique_sources = sorted(df['source'].unique())

# Create 2x3 subplot figure (maintaining original 7x3 aspect ratio for each subplot)
fig, axes = plt.subplots(2, 3, figsize=(16.3, 5.3))
axes = axes.flatten()

# Plot 1: First-stage accuracy vs diff1
per1 = df.groupby(['source', 'wid', 'diff1'])['correct1'].mean().reset_index(name='accuracy1')
stats = per1.groupby(['source', 'diff1'])['accuracy1'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

for source, grp in stats.groupby('source'):
    color = get_source_color(source)
    axes[0].errorbar(grp['diff1'], grp['mean'], yerr=grp['sem'],
                    marker='o', linewidth=1.5, capsize=4, color=color)
axes[0].set_xlabel('Max value - mean other value', fontsize=17)
axes[0].set_ylabel('1st-stage accuracy', fontsize=17)
# axes[0].set_title('A. First-stage accuracy')

# Plot 2: Second-stage accuracy vs diff2
per2 = df.groupby(['source', 'wid', 'diff2'])['correct2'].mean().reset_index(name='accuracy2')
stats = per2.groupby(['source', 'diff2'])['accuracy2'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

for source, grp in stats.groupby('source'):
    color = get_source_color(source)
    axes[1].errorbar(grp['diff2'], grp['mean'], yerr=grp['sem'],
                     marker='o', linewidth=1.5, capsize=4, color=color)
axes[1].set_xlabel('Absolute reward difference', fontsize=17)
axes[1].set_ylabel('2nd-stage accuracy', fontsize=17)
# axes[1].set_title('B. Second-stage accuracy')

# Plot 3: Overall accuracy vs diff1
per_all = df.groupby(['source', 'wid', 'diff1'])['correct_all'].mean().reset_index(name='accuracy')
stats = per_all.groupby(['source', 'diff1'])['accuracy'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

for source, grp in stats.groupby('source'):
    color = get_source_color(source)
    axes[2].errorbar(grp['diff1'], grp['mean'], yerr=grp['sem'],
                     marker='o', linewidth=1.5, capsize=4, color=color)
axes[2].set_xlabel('Max value - mean other value', fontsize=17)
axes[2].set_ylabel('Overall accuracy', fontsize=17)
# axes[2].set_title('C. Overall accuracy')

# Plot 4: First-stage RT vs diff1
rt1 = df[df['correct1']].groupby(['source', 'wid', 'diff1'])['rt1'].mean().reset_index(name='mean_rt1')
stats = rt1.groupby(['source', 'diff1'])['mean_rt1'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

for source, grp in stats.groupby('source'):
    color = get_source_color(source)
    axes[3].errorbar(grp['diff1'], grp['mean'], yerr=grp['sem'],
                     marker='o', linewidth=1.5, capsize=4, color=color)
axes[3].set_xlabel('Max value - mean other value', fontsize=17)
axes[3].set_ylabel('1st-stage RT (ms)', fontsize=17)
# axes[3].set_title('D. First-stage RT')

# Plot 5: Second-stage RT vs diff2
rt2 = df[df['correct2']].groupby(['source', 'wid', 'diff2'])['rt2'].mean().reset_index(name='mean_rt2')
stats = rt2.groupby(['source', 'diff2'])['mean_rt2'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])

for source, grp in stats.groupby('source'):
    color = get_source_color(source)
    axes[4].errorbar(grp['diff2'], grp['mean'], yerr=grp['sem'],
                     marker='o', linewidth=1.5, capsize=4, color=color)
axes[4].set_xlabel('Absolute reward difference', fontsize=17)
axes[4].set_ylabel('2nd-stage RT (ms)', fontsize=17)
# axes[4].set_title('E. Second-stage RT')

# Plot 6: First-stage RT vs subtree_relation
stats = df.groupby(['source', 'subtree_relation'])['rt1'].agg(['mean', 'sem']).reset_index()

for source, grp in stats.groupby('source'):
    color = get_source_color(source)
    axes[5].errorbar(grp['subtree_relation'], grp['mean'], yerr=grp['sem'],
                     marker='o', linewidth=1.5, capsize=4, color=color)
axes[5].set_xlabel('Tree configuration', fontsize=17)
axes[5].set_xticks([1, 2, 3])
axes[5].set_xticklabels(['Max &\n2nd-best', 'Max &\n3rd-best', 'Max &\nMin'])
axes[5].set_ylabel('1st-stage RT (ms)', fontsize=17)
#  axes[5].set_title('F. First-stage RT by tree configuration')

# Create legend at bottom, one row
legend_elements = [
    Line2D([0], [0], color=get_source_color(source), marker='o',
           label=str(source), linestyle='-', linewidth=2)
    for source in unique_sources
]

# Adjust layout to leave space for legend at bottom (smaller gap)
plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=3.0, w_pad=2.0)

# Add legend at bottom, horizontal row, closer to subplots
fig.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 0.05), ncol=len(unique_sources), frameon=True, fontsize=13)

plt.savefig('Tree2/figures/Figure4_2.svg', bbox_inches='tight')
plt.show()
