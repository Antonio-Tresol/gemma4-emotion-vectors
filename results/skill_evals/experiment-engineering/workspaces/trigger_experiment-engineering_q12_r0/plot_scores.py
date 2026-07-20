import json
import matplotlib.pyplot as plt
import numpy as np

# Read JSONL file
data = []
with open('results/scores.jsonl', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# Calculate accuracy percentages and organize by model and task
models = {}
for entry in data:
    model = entry['model']
    task = entry['task']
    accuracy = (entry['correct'] / entry['total']) * 100

    if model not in models:
        models[model] = {}
    models[model][task] = accuracy

# Prepare data for grouped bar chart
tasks = list(models[list(models.keys())[0]].keys())
model_names = list(models.keys())

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for each model
colors = ['#2E86AB', '#A23B72']  # Blue and purple for categorical identity
for i, model in enumerate(model_names):
    accuracies = [models[model][task] for task in tasks]
    ax.bar(x + i*width, accuracies, width, label=model.capitalize(), color=colors[i])

# Customize chart
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_xlabel('Task', fontsize=11)
ax.set_title('Model Accuracy by Task', fontsize=13, fontweight='bold')
ax.set_xticks(x + width / 2)
ax.set_xticklabels([task.capitalize() for task in tasks])
ax.legend()
ax.set_ylim(0, 100)

# Add grid for readability
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add value labels on bars
for i, model in enumerate(model_names):
    accuracies = [models[model][task] for task in tasks]
    for j, acc in enumerate(accuracies):
        ax.text(j + i*width, acc + 1.5, f'{acc:.0f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/scores_chart.png', dpi=300, bbox_inches='tight')
print('Chart saved to results/scores_chart.png')
plt.show()
