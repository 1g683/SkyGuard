import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.family'] = 'STIXGeneral'
try:
    TN = int(input("Enter TN (True Negative: actually normal, predicted normal): "))
    FP = int(input("Enter FP (False Positive: actually normal, predicted spoofed): "))
    FN = int(input("Enter FN (False Negative: actually spoofed, predicted normal): "))
    TP = int(input("Enter TP (True Positive: actually spoofed, predicted spoofed): "))
except ValueError:
    print("Invalid input. Please enter integers only.")
    exit()

cm = np.array([[TN, FP], [FN, TP]])

# set lable
labels = ['Normal', 'GPS Spoofed']

# draw
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 12}, cbar_kws={'shrink': 0.8})
cbar= plt.gcf().axes[-1]  
cbar.tick_params(labelsize=12)

ax = plt.gca()
ax.tick_params(axis='x', labelsize=14) 
ax.tick_params(axis='y', labelsize=14)

plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.tight_layout()
plt.savefig('confusion_matrix.pdf', dpi=600)
plt.show()
