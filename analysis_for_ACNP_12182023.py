# Part5: Evaluate the performance of the algorithms.


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Initialize a 5x2 confusion matrix
# total_confusion = np.zeros((5, 2))

# Given confusion matrix data
total_confusion = np.array([[39276, 4807], [923, 1794]])


# Dictionaries for label and prediction mappings
prediction_mapping = {0: 'None', 1: 'Crying'}
groundtruth_mapping = {0: 'None', 1: 'Cry'}

# Normalize the confusion matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Normalize by row
row_sums = total_confusion.sum(axis=1)
normalized_confusion = total_confusion / row_sums[:, np.newaxis]

# Create folder if it doesn't exist
if not os.path.exists('analysis'):
    os.mkdir('analysis')

# Plot the heatmap
plt.figure(figsize=(10, 7))
ax = sns.heatmap(normalized_confusion, annot=True, cmap="YlGnBu", xticklabels=list(prediction_mapping.values()), yticklabels=list(groundtruth_mapping.values()), fmt=".4%", linewidths=1, linecolor='gray')

# Adding counts to cells
for i, j in itertools.product(range(normalized_confusion.shape[0]), range(normalized_confusion.shape[1])):
    count = total_confusion[i, j]
    plt.text(j + 0.5, i + 0.7, f'\n({int(count)})', ha='center', va='center', color='red', fontsize=15)

# Adjusting labels and title
ax.set_xlabel('Predicted')
ax.set_ylabel('Ground Truth')
ax.set_title('Normalized Confusion Matrix by Row')
plt.yticks(va="center")
plt.tight_layout()

# Saving the figure
save_path = os.path.join('analysis', 'analysis-12182023', 'normalized_confusion_matrix.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Given confusion matrix data
total_confusion = np.array([[39276, 4807], [923, 1794]])

# Flatten the confusion matrix for metric calculations
true_negatives, false_positives, false_negatives, true_positives = total_confusion.flatten()

# Calculating metrics
accuracy = (true_positives + true_negatives) / np.sum(total_confusion)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
f1 = 2 * (precision * recall) / (precision + recall)

# Additional metrics
# True Negative Rate (TNR) or Specificity
tnr = true_negatives / (true_negatives + false_positives)
# False Positive Rate (FPR)
fpr = false_positives / (false_positives + true_negatives)
# False Negative Rate (FNR)
fnr = false_negatives / (false_negatives + true_positives)
# Positive Predictive Value (PPV) or Precision
ppv = precision
# Negative Predictive Value (NPV)
npv = true_negatives / (true_negatives + false_negatives)
# False Discovery Rate (FDR)
fdr = false_positives / (false_positives + true_positives)

accuracy, precision, recall, specificity, f1, tnr, fpr, fnr, ppv, npv, fdr

