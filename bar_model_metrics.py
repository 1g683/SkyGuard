import matplotlib.pyplot as plt
import numpy as np

model_names = ['Siamese ViTNet (ours)', 'Siamese ResNet', 'Siamese SqueezeNet', 'Siamese EfficientNet', 'Siamese MobileNet']
plt.rcParams['font.family'] = 'STIXGeneral'

# data
train_metrics = np.array([
    [0.9302, 0.9190, 0.9306, 0.9248],
    [0.8910, 0.8855, 0.9031, 0.8942],
    [0.8561, 0.8405, 0.8536, 0.8470],
    [0.8445, 0.8050, 0.8865, 0.8438],
    [0.8227, 0.8046, 0.8719, 0.8369]
])

val_metrics = np.array([
    [0.9070, 0.8936, 0.9333, 0.9130],
    [0.8488, 0.8210, 0.8966, 0.8571],
    [0.8605, 0.8028, 0.8507, 0.8261],
    [0.8314, 0.8023, 0.8519, 0.8263],
    [0.8081, 0.8043, 0.8315, 0.8177]
])

test_metrics = np.array([
    [0.8411, 0.8361, 0.8793, 0.8571],
    [0.7757, 0.7500, 0.9000, 0.8182],
    [0.7103, 0.7215, 0.8636, 0.7861],
    [0.8131, 0.7292, 0.8333, 0.7778],
    [0.7477, 0.7091, 0.8302, 0.7652]
])

def plot_metric_bar(metrics, title, save_path):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    x = np.arange(len(metric_names))
    width = 0.12
    num_models = metrics.shape[0]

    # Color and Fill Style
    colors = ['#5E519B', '#3B8DB3', '#76C6A5', '#b1d85c', '#FAE8A2']
    hatches = ['//', '\\\\', '||', '--', '++']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(num_models):
        offset = (i - num_models / 2) * width + width / 2
        ax.bar(x + offset, metrics[i],
               width,
               label=model_names[i],
               color=colors[i % len(colors)],
               hatch=hatches[i % len(hatches)],
               edgecolor='black',
               linewidth=1.0,
               alpha=0.9)

    ax.set_xlabel('Metrics', fontsize=26)
    ax.set_ylabel('Score', fontsize=26)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=26)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

# draw
plot_metric_bar(train_metrics, '', 'bar_train_metrics.pdf')
plot_metric_bar(val_metrics, '', 'bar_val_metrics.pdf')
plot_metric_bar(test_metrics, '', 'bar_test_metrics.pdf')
