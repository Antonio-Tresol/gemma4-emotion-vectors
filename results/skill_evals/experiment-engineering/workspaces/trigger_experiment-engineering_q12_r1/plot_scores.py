import json
import matplotlib.pyplot as plt
import numpy as np

# Read scores.jsonl
scores = []
with open('results/scores.jsonl') as f:
    for line in f:
        if line.strip():
            scores.append(json.loads(line))

# Organize data
data_by_task = {}
for row in scores:
    task = row['task']
    if task not in data_by_task:
        data_by_task[task] = {}
    data_by_task[task][row['model']] = row['correct']

tasks = sorted(data_by_task.keys())
models = sorted(set(row['model'] for row in scores))

# Prepare arrays
x = np.arange(len(tasks))
width = 0.35
colors = ['#1f77b4', '#ff7f0e']  # Blue (haiku), Orange (sonnet)

fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars for each model
for i, model in enumerate(models):
    values = [data_by_task[task].get(model, 0) for task in tasks]
    ax.bar(x + i * width, values, width, label=model.capitalize(), color=colors[i], alpha=0.8)

# Customize
ax.set_xlabel('Task', fontsize=11, fontweight='bold')
ax.set_ylabel('Correct Answers (out of 50)', fontsize=11, fontweight='bold')
ax.set_title('Model Performance by Task', fontsize=13, fontweight='bold')
ax.set_xticks(x + width / 2)
ax.set_xticklabels([t.capitalize() for t in tasks])
ax.set_ylim(0, 50)
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)
ax.legend()

plt.tight_layout()
plt.savefig('results/scores_chart.png', dpi=150, bbox_inches='tight')
print("Chart saved to results/scores_chart.png")
plt.show()
