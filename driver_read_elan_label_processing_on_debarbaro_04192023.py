# Wed Apr 19 11:37:22 EDT 2023 by Kyunghun Lee (Created)

#  a script was created to process ELAN-generated labels and produce output
#  suitable for follow-up machine learning analysis using Yao's pre-trained
#  model. The input file consists of LENA labels for 37 infants at 12 months
#  and is 10 minutes in length. The input labels required by the script can be
#  located at "/Volumes/NNT/Lauren Henry Projects/LENA project/Final text files
#  for data analysis" or in the MS Team at "data/FINAL text files for data
#  analysis_03302023.zip".0

# Model: Yao's pre-trained model
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
from preprocessing import preprocessing
from predict import predict
for inFolder in inFolders:
    preprocessedFolder = inFolder + '/preprocessed/'
    predictedFolder = inFolder + '/predicted/'

    # create the output folder if it does not exist
    if not os.path.exists(preprocessedFolder):
        os.makedirs(preprocessedFolder)
    if not os.path.exists(predictedFolder):
        os.makedirs(predictedFolder)

    # Sort the files by their number
    inFiles = sorted(glob.glob(os.path.join(inFolder + '/combined*.wav')), key=sort_key)

    for inFile in inFiles:
        preprocessedFile = preprocessedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
        predictedFile = predictedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
        scoreFile = 'test.csv'
        # Run Preproecessing
        preprocessing(inFile, preprocessedFile)

        # Run Prediction script
        _,pred_prob = predict(inFile, preprocessedFile, predictedFile,scoreFile)
        # audio_filename,preprocessed_file,output_file,prob_file=None


# Part 3: Evaluate the performance of the algorithms.

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
home = expanduser("~")
inFolders = glob.glob(home + "/data/processed/deBarbaroCry_2min/P*")
inFolders.sort()
predictionFiles = []
labelFiles = []
for inFolder in inFolders:
    predictedFolder = inFolder + '/predicted/'
    # labelFolder = home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan + '/groundTruth/'
    labelFolder =  inFolder + '/groundTruth/'
    predictionFiles += sorted(glob.glob(os.path.join(predictedFolder + "*.csv")), key=sort_key)
    labelFiles += sorted(glob.glob(os.path.join(labelFolder + "*.csv")), key=sort_key)

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

# Part4: Analysis

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