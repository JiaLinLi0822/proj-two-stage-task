import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import json
import pandas as pd

with open("data/Tree2_v3.json", "r") as f_real:
    real_data = [json.loads(line) for line in f_real]
for entry in real_data:
    entry["source"] = "real"

# # Step 2: Load simulated data
# with open("results/Tree2/v3/simulated_v3_model1.json", "r") as f_sim:
#     model1_data = [json.loads(line) for line in f_sim]
# for entry in model1_data:
#     entry["source"] = "model1"

# with open("results/Tree2/v3/simulated_v3_model2.json", "r") as f_sim:
#     model2_data = [json.loads(line) for line in f_sim]
# for entry in model2_data:
#     entry["source"] = "model2"

# with open("results/Tree2/v3/simulated_v3_model3.json", "r") as f_sim:
#     model3_data = [json.loads(line) for line in f_sim]
# for entry in model3_data:
#     entry["source"] = "model3"

# with open("results/Tree2/v3/simulated_v3_model4.json", "r") as f_sim:
#     model4_data = [json.loads(line) for line in f_sim]
# for entry in model4_data:
#     entry["source"] = "model4"

# with open("results/Tree2/v3/simulated_v3_model5.json", "r") as f_sim:
#     model5_data = [json.loads(line) for line in f_sim]
# for entry in model5_data:
#     entry["source"] = "model5"

with open("results/Tree2/v3/simulated_v3_model6.json", "r") as f_sim:
    model6_data = [json.loads(line) for line in f_sim]
for entry in model6_data:
    entry["source"] = "model6"

with open("results/Tree2/v3/simulated_v3_model7.json", "r") as f_sim:
    model7_data = [json.loads(line) for line in f_sim]
for entry in model7_data:
    entry["source"] = "model7"

with open("results/Tree2/v3/simulated_v3_model8.json", "r") as f_sim:
    model8_data = [json.loads(line) for line in f_sim]
for entry in model8_data:
    entry["source"] = "model8"

with open("results/Tree2/v3/simulated_v3_model9.json", "r") as f_sim:
    model9_data = [json.loads(line) for line in f_sim]
for entry in model9_data:
    entry["source"] = "model9"

with open("results/Tree2/v3/simulated_v3_model10.json", "r") as f_sim:
    model10_data = [json.loads(line) for line in f_sim]
for entry in model10_data:
    entry["source"] = "model10"

# df = real_data + model1_data + model2_data + model3_data + model4_data + model5_data
df = real_data + model6_data + model7_data + model8_data + model9_data + model10_data
df = pd.DataFrame(df)


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
    

# --- 通用 colormap 设置 ---
unique_sources = sorted(df['source'].unique())
n_sources = len(unique_sources)
cmap = plt.get_cmap('viridis').resampled(n_sources)
norm = mcolors.Normalize(vmin=0, vmax=n_sources - 1)
legend_elements = []

# first stage accuracy
df['best_path_idx'] = df['rewards'].apply(lambda v: int(np.argmax(v)))
df['correct1'] = df.apply(
    lambda r: (r['best_path_idx'] < 2 and r['choice1'] == 1)
              or (r['best_path_idx'] >= 2 and r['choice1'] == 2),
    axis=1
)

# second stage accuracy
df['correct2'] = df.apply(
    lambda r: (
        r['value2'][r['choice2'] - 1]
        == max(subtree_vals(r['value2'], r['choice1']))
    ),
    axis=1
)
df['correct_all'] = df['correct1'] & df['correct2']

# 1. 第一阶段 Accuracy 统计
per1 = (
    df.groupby(['source','wid','diff1'])['correct1']
      .mean()
      .reset_index(name='accuracy1')
)
stats1 = (
    per1.groupby(['source','diff1'])['accuracy1']
        .agg(['mean','std','count'])
        .reset_index()
)
stats1['sem'] = stats1['std'] / np.sqrt(stats1['count'])

# 2. 第二阶段 Accuracy 统计
per2 = (
    df.groupby(['source','wid','diff2'])['correct2']
      .mean()
      .reset_index(name='accuracy2')
)
stats2 = (
    per2.groupby(['source','diff2'])['accuracy2']
         .agg(['mean','std','count'])
         .reset_index()
)
stats2['sem'] = stats2['std'] / np.sqrt(stats2['count'])

# 3. 第一阶段 RT 统计（仅正确 trial）
rt1 = (
    df[df['correct1']]
      .groupby(['source','wid','diff1'])['rt1']
      .mean()
      .reset_index(name='mean_rt1')
)
stats3 = (
    rt1.groupby(['source','diff1'])['mean_rt1']
        .agg(['mean','std','count'])
        .reset_index()
)
stats3['sem'] = stats3['std'] / np.sqrt(stats3['count'])

# 4. 第二阶段 RT 统计（仅正确 trial）
rt2 = (
    df[df['correct2']]
      .groupby(['source','wid','diff2'])['rt2']
      .mean()
      .reset_index(name='mean_rt2')
)
stats4 = (
    rt2.groupby(['source','diff2'])['mean_rt2']
        .agg(['mean','std','count'])
        .reset_index()
)
stats4['sem'] = stats4['std'] / np.sqrt(stats4['count'])

# 5. 整体 Accuracy 统计
per_all = (
    df.groupby(['source','wid','diff1'])['correct_all']
      .mean()
      .reset_index(name='accuracy')
)
stats5 = (
    per_all.groupby(['source','diff1'])['accuracy']
         .agg(['mean','std','count'])
         .reset_index()
)
stats5['sem'] = stats5['std'] / np.sqrt(stats5['count'])



df['subtree_relation'] = df['rewards'].apply(subtree_relation_code)
stats6 = df.groupby(['source','subtree_relation'])['rt1'].agg(['mean','sem']).reset_index()

# === 创建 2x3 子图 ===
fig, axes = plt.subplots(2, 3, figsize=(18, 6))
axes = axes.flatten()

# 图 1: 第一阶段 Accuracy
ax = axes[0]
for idx, (source, grp) in enumerate(stats1.groupby('source')):
    color = cmap(norm(idx))
    ax.errorbar(grp['diff1'], grp['mean'], yerr=grp['sem'],
                marker='o', linewidth=2, capsize=5, color=color)
    if idx < n_sources:
        legend_elements.append(Line2D([0],[0], color=color, marker='o',
                                       linestyle='-', linewidth=2, label=source))
ax.set_title('First-stage Accuracy')
ax.set_xlabel('Max value - mean other value')
ax.set_ylabel('Accuracy')

# 图 2: 第二阶段 Accuracy
ax = axes[1]
for idx, (source, grp) in enumerate(stats2.groupby('source')):
    color = cmap(norm(idx))
    ax.errorbar(grp['diff2'], grp['mean'], yerr=grp['sem'],
                marker='o', linewidth=2, capsize=5, color=color)
ax.set_title('Second-stage Accuracy')
ax.set_xlabel('Absolute reward difference')
ax.set_ylabel('Accuracy')

# 图 3: 第一阶段 RT
ax = axes[3]
for idx, (source, grp) in enumerate(stats3.groupby('source')):
    color = cmap(norm(idx))
    ax.errorbar(grp['diff1'], grp['mean'], yerr=grp['sem'],
                marker='o', linewidth=2, capsize=5, color=color)
ax.set_title('First-stage RT (ms)')
ax.set_xlabel('Max value - mean other value')
ax.set_ylabel('RT (ms)')

# 图 4: 第二阶段 RT
ax = axes[4]
for idx, (source, grp) in enumerate(stats4.groupby('source')):
    color = cmap(norm(idx))
    ax.errorbar(grp['diff2'], grp['mean'], yerr=grp['sem'],
                marker='o', linewidth=2, capsize=5, color=color)
ax.set_title('Second-stage RT (ms)')
ax.set_xlabel('Absolute reward difference')
ax.set_ylabel('RT (ms)')

# 图 5: 整体 Accuracy
ax = axes[5]
for idx, (source, grp) in enumerate(stats5.groupby('source')):
    color = cmap(norm(idx))
    ax.errorbar(grp['diff1'], grp['mean'], yerr=grp['sem'],
                marker='o', linewidth=2, capsize=5, color=color)
ax.set_title('Overall Accuracy')
ax.set_xlabel('Max value - mean other value')
ax.set_ylabel('Accuracy')

# 图 6: Subtree Relation RT
ax = axes[2]
for idx, (source, grp) in enumerate(stats6.groupby('source')):
    color = cmap(norm(idx))
    ax.errorbar(grp['subtree_relation'], grp['mean'], yerr=grp['sem'],
                marker='o', linewidth=2, capsize=5, color=color)
ax.set_title('First-stage RT by Subtree Relation')
ax.set_xlabel('Tree configuration')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Max & 2nd-best', 'Max & 3rd-best', 'Max & Min'])
ax.set_ylabel('RT (ms)')

# 添加共享图注于上方
fig.legend(handles=legend_elements, loc='upper center', ncol=n_sources, fontsize=12)
# fig.legend(
#     handles=legend_elements,
#     loc='center left',
#     bbox_to_anchor=(1, 0.5),
#     ncol=1,
#     fontsize=12
# )
fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()