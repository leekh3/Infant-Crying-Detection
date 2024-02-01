# Model: Yao's training on yao's dataset, but using our effort.
# using alex_svm_kyunghun.py, train_alex_kyunghun.py, prediction_kyunghun.py
# training dataset: deBarbaro
# Target: deBarbaro

from os.path import expanduser
import glob
import re
import pandas as pd
import os
from pydub import AudioSegment

def printTime(df_tmp):
    df_tmp['beginTime'] = df_tmp['Begin Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    df_tmp['endTime'] = df_tmp['End Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    print(df_tmp.loc[:, ['ID','Type','beginTime', 'endTime']])

# Part1: Combine multiple 5-second .wav files into 2-minute audio files and generate corresponding label files with 5
# repeated labels per input file based on the presence of 'cry' or 'notcry' in the input file name.

# Concatenation (2min)
home = expanduser("~")
# find all folders
inFolders = glob.glob(home + "/data/deBarbaroCry/P*")
inFolders.sort()
inFolders = inFolders[:2]

def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1

for inFolder in inFolders:
    subFolder = inFolder.split('/')[-1]
    outFolder = home + "/data/processed/deBarbaroCry_2min/" + subFolder + '/'
    labelFolder = home + "/data/processed/deBarbaroCry_2min/" + subFolder + '/groundTruth'
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
        print(f"Directory '{outFolder}' created.")
    else:
        print(f"Directory '{outFolder}' already exists.")

    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)
        print(f"Directory '{labelFolder}' created.")
    else:
        print(f"Directory '{labelFolder}' already exists.")

    # Sort the files by their number
    wav_files = sorted(glob.glob(os.path.join(inFolder, subFolder + '_*.wav')), key=sort_key)

    # Set the maximum duration for each combined file (2 minutes in this case)
    max_duration = 2 * 60 * 1000  # milliseconds

    combined_audio = AudioSegment.empty()
    file_counter = 0

    labels = []
    for wav_file in wav_files:
        audio = AudioSegment.from_wav(wav_file)
        combined_audio += audio

        # Add label based on 'cry' or 'notcry' in file name
        if 'notcry' in wav_file:
            labels.extend([0] * 5)
        elif 'cry' in wav_file:
            labels.extend([1] * 5)

        if combined_audio.duration_seconds >= 120 or wav_file == wav_files[-1]:
            output_file = os.path.join(outFolder, f"combined_file_{file_counter}.wav")
            combined_audio.export(output_file, format="wav")

            label_file = os.path.join(labelFolder, f"labels_{file_counter}.csv")
            with open(label_file, 'w') as f:
                f.write('\n'.join(map(str, labels)))

            file_counter += 1
            combined_audio = AudioSegment.empty()
            labels = []

# Part2: Run Yao's preprocessing/prediction script (refer: previous my code:
# driver_baseline_concatenate_deBarbaroCry_2min.py)
home = expanduser("~")
inFolders = glob.glob(home + "/data/processed/deBarbaroCry_2min/P*")
inFolders.sort()
inFolders = inFolders[:2]

from preprocessing import preprocessing
from predict_kyunghun import predict_kyunghun
for inFolder in inFolders:
    preprocessedFolder = inFolder + '/preprocessed/'
    predictedFolder = inFolder + '/predicted/'
    probFolder = inFolder + '/prob/'

    # create the output folder if it does not exist
    if not os.path.exists(preprocessedFolder):
        os.makedirs(preprocessedFolder)
    if not os.path.exists(predictedFolder):
        os.makedirs(predictedFolder)
    if not os.path.exists(probFolder):
        os.makedirs(probFolder)

    # Sort the files by their number
    inFiles = sorted(glob.glob(os.path.join(inFolder + '/combined*.wav')), key=sort_key)

    for inFile in inFiles:
        preprocessedFile = preprocessedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
        predictedFile = predictedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
        probFile = probFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
        # Run Preproecessing
        preprocessing(inFile, preprocessedFile)

        # Run Prediction script
        # inFile = '/home/leek13/data/processed/deBarbaroCry_2min/P06/combined_file_0.wav'
        # preprocessedFile = '/home/leek13/data/processed/deBarbaroCry_2min/P06/preprocessed/0.csv'
        # predictedFile = '/home/leek13/data/processed/deBarbaroCry_2min/P06/predicted/0.csv'
        # probFile = '/home/leek13/data/processed/deBarbaroCry_2min/P06/prob/0.csv'

        _,pred_prob = predict(inFile, preprocessedFile, predictedFile,probFile,'./trained/pics_alex_noflip_torch_distress.h5')
        # audio_filename = inFile;preprocessed_file=preprocessedFile;output_file=predictedFile;prob_file = probFile;

        # predict(audio_filename, preprocessed_file, output_file, prob_file=None):
        # audio_filename,preprocessed_file,output_file,prob_file=None


# Part 3: Evaluate the performance of the algorithms.

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
home = expanduser("~")
inFolders = glob.glob(home + "/data/processed/deBarbaroCry_2min/P*")
inFolders.sort()
predictionFiles = []
labelFiles = []
probFiles = []
for inFolder in inFolders:
    predictedFolder = inFolder + '/predicted/'
    # labelFolder = home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan + '/groundTruth/'
    labelFolder =  inFolder + '/groundTruth/'
    probFolder = inFolder + '/prob/'
    predictionFiles += sorted(glob.glob(os.path.join(predictedFolder + "*.csv")), key=sort_key)
    labelFiles += sorted(glob.glob(os.path.join(labelFolder + "*.csv")), key=sort_key)
    probFiles += sorted(glob.glob(os.path.join(probFolder + "*.csv")), key=sort_key)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

total_confusion_matrix = None

# Loop through each file pair
for prediction_file, label_file in zip(predictionFiles, labelFiles):
    # Read the files
    pred_df = pd.read_csv(prediction_file, header=None, names=['Label'])
    label_df = pd.read_csv(label_file, header=None, names=['Label'])

    # Calculate the confusion matrix for the current file pair
    current_confusion_matrix = confusion_matrix(label_df['Label'], pred_df['Label'])

    # Add the current confusion matrix to the total confusion matrix
    if total_confusion_matrix is None:
        total_confusion_matrix = current_confusion_matrix
    else:
        total_confusion_matrix += current_confusion_matrix


import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np

# Normalize by row
total_confusion = total_confusion_matrix
row_sums = total_confusion.sum(axis=1)
normalized_confusion = total_confusion / row_sums[:, np.newaxis]

# Define mappings for the 2x2 case
prediction_mapping = {0: 'None', 1: 'Crying'}
groundtruth_mapping = {0: 'None', 1: 'Crying'}

# Plot the heatmap
plt.figure(figsize=(10, 7))
ax = sns.heatmap(normalized_confusion, annot=True, cmap="YlGnBu", xticklabels=list(prediction_mapping.values()), yticklabels=list(groundtruth_mapping.values()), fmt=".4%", linewidths=1, linecolor='gray')

# Adding counts to cells
for i, j in itertools.product(range(normalized_confusion.shape[0]), range(normalized_confusion.shape[1])):
    count = total_confusion[i, j]
    plt.text(j + 0.5, i + 0.6, f'\n({int(count)})', ha='center', va='center', color='red', fontsize=15)

# Adjusting labels and title
ax.set_xlabel('Predicted')
ax.set_ylabel('Ground Truth')
ax.set_title('Normalized Confusion Matrix by Row')
plt.yticks(va="center")
plt.tight_layout()

# Saving the figure
# save_path = os.path.join('analysis', 'analysis-10042023', '2x2_normalized_confusion_matrix.png')
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# plt.savefig(save_path, bbox_inches='tight')
plt.show()








# Part4: Analysis I (based on confusion matrix)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

confusion_mat = total_confusion_matrix

# Calculate evaluation metrics
def calculate_metrics(confusion_mat):
    true_pos = np.diag(confusion_mat)
    false_pos = np.sum(confusion_mat, axis=0) - true_pos
    false_neg = np.sum(confusion_mat, axis=1) - true_pos
    true_neg = np.sum(confusion_mat) - (true_pos + false_pos + false_neg)

    accuracy = np.sum(true_pos) / np.sum(confusion_mat)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

accuracy, precision, recall, f1_score = calculate_metrics(confusion_mat)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1_score}")

# Create a heatmap
def plot_heatmap(confusion_mat):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues', annot_kws={"size": 12})
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_heatmap(confusion_mat)

# Part4: Analysis II (based on ROC curves)
from sklearn.metrics import roc_curve, auc

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
home = expanduser("~")
inFolders = glob.glob(home + "/data/processed/deBarbaroCry_2min/P*")
inFolders.sort()
labelFiles = []
probFiles = []
for inFolder in inFolders:
    labelFolder =  inFolder + '/groundTruth/'
    probFolder = inFolder + '/prob/'
    labelFiles += sorted(glob.glob(os.path.join(labelFolder + "*.csv")), key=sort_key)
    probFiles += sorted(glob.glob(os.path.join(probFolder + "*.csv")), key=sort_key)

# Read data and concatenate them
probs, labels = [], []
for probFile, labelFile in zip(probFiles, labelFiles):
    prob_df = pd.read_csv(probFile, header=None)
    label_df = pd.read_csv(labelFile, header=None)

    probs.extend(prob_df.values.flatten())
    labels.extend(label_df.values.flatten())

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()