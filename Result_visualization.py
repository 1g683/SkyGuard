import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
plt.rcParams['font.family'] = 'STIXGeneral'
model_name = input("Enter the model name to load visualization data: ")
filename = f'vis_data_{model_name}.pkl'
with open(filename, 'rb') as f:
    pred_matrix, output_matrix, label_matrix = pickle.load(f)

def plot_heatmap(data_split, save_name):
    output = output_matrix[data_split][:50].reshape(1, -1)
    pred = pred_matrix[data_split][:50].reshape(1, -1)
    label = label_matrix[data_split][:50].reshape(1, -1)

    all_data = np.vstack([output, pred, label])  # shape: (3, 50)

    fig, ax = plt.subplots(figsize=(25, 3))

    im_list = []
    for i in range(3):
        row = all_data[i].reshape(1, -1)
        cmap = plt.get_cmap("YlGnBu")
        norm = None  

        im=ax.imshow(row, aspect='auto', extent=[0, 50, 2 - i, 3 - i], cmap=cmap, norm=norm)
        im_list.append(im)

    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['true label', 'prediction', 'output'], fontsize=20)
    ax.set_xticks(np.arange(0.5, 50, 1))
    ax.set_xticklabels(np.arange(1, 51), fontsize=20)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 3)

    ax.set_xlabel('Sample Index', fontsize=24)
    ax.tick_params(axis='x', which='major', labelsize=20)  
    ax.tick_params(axis='y', which='major', labelsize=24)
    ax.tick_params(axis='x', labelrotation=90)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im_list[2], ax=ax, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=20)
    cbar.outline.set_visible(False)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0', '1'])
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

plot_heatmap('train', 'Result_visualization_train.png')
plot_heatmap('val', 'Result_visualization_val.png')
