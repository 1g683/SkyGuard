import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'STIXGeneral'

data = {
    'Siamese ViTNet (ours)': [0.9070, 0.8936, 0.9333, 0.9130],
    'Siamese ResNet':        [0.8488, 0.8210, 0.8966, 0.8571],
    'Siamese SqueezeNet':    [0.8605, 0.8028, 0.8507, 0.8261],
    'Siamese EfficientNet':  [0.8314, 0.8023, 0.8519, 0.8263],
    'Siamese MobileNet':     [0.8081, 0.8043, 0.8315, 0.8177],
}

labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
num_vars = len(labels)

# Angle division
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# closed figure
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

colors = ['#5E519B', '#3B8DB3', '#76C6A5', '#b1d85c', '#FAE8A2']

# Draw each model curve
def normalize(scores, min_val=0.7, max_val=1.0):
    return [(s - min_val) / (max_val - min_val) for s in scores]

for i, (model, scores) in enumerate(data.items()):
    norm_scores = normalize(scores)
    values = norm_scores + norm_scores[:1]
    ax.plot(angles, values, label=model, color=colors[i], linewidth=2, marker='o', markersize=5, zorder=2)
    ax.fill(angles, values, color=colors[i], alpha=0.1, zorder=1)

# Set coordinate labels
ax.set_theta_offset(np.pi / 2 )
ax.set_theta_direction(-1)
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.7', '0.775', '0.85', '0.925', ''], color='gray', fontsize=16)
ax.set_rlabel_position(0)
ax.tick_params(colors='gray')
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=20, color='black')
ax.spines['polar'].set_color('gray')
ax.spines['polar'].set_linewidth(1)
ax.grid(color='lightgray', linestyle='-', linewidth=0.8, zorder=3)

# Set coordinate axis range
ax.set_ylim(0, 1)

plt.legend(loc='upper right', fontsize=18, bbox_to_anchor=(1.30, 1.05))

plt.tight_layout()
plt.savefig('plot_model_radar.pdf', dpi=600)
plt.show()