import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = []
with open('results/scores.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Organize data
models = sorted(set(d['model'] for d in data))
tasks = sorted(set(d['task'] for d in data))

# Create matrix of accuracies
accuracies = {}
for d in data:
    key = (d['model'], d['task'])
    accuracies[key] = d['correct'] / d['total']

# Setup plot
x = np.arange(len(tasks))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each model
for i, model in enumerate(models):
    values = [accuracies.get((model, task), 0) for task in tasks]
    ax.bar(x + i * width, values, width, label=model)

# Customize
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Task', fontsize=12)
ax.set_title('Model Performance by Task', fontsize=14, fontweight='bold')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(tasks)
ax.legend()
ax.set_ylim([0, 1])

# Add value labels on bars
for i, model in enumerate(models):
    values = [accuracies.get((model, task), 0) for task in tasks]
    for j, v in enumerate(values):
        ax.text(j + i * width, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/scores.png', dpi=300, bbox_inches='tight')
print("Chart saved to results/scores.png")
plt.show()
