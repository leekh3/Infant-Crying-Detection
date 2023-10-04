# ---------------------------------------------------------
# Script Information:
# Creation Date: Thu Mar 30 09:59:45 EDT 2023
# Author: Kyunghun Lee
# Description: Process ELAN-generated labels for machine learning analysis using Yao's pre-trained model.
# Input: LENA labels for 37 infants at 12 months, each 10 minutes long.
# Location: /Volumes/NNT/Lauren Henry Projects/LENA project/Final text files for data analysis
#           or data/FINAL text files for data analysis_03302023.zip
# Model: Yao's pre-trained model
# Dataset: LENA (100 subjects, each 10min long)
# ---------------------------------------------------------

from os.path import expanduser
import glob
import re
import pandas as pd
import os

# Helper function to extract and sort by numbers from filenames
def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1

# Function to format time columns and print relevant information
def printTime(df_tmp):
    df_tmp['beginTime'] = df_tmp['Begin Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    df_tmp['endTime'] = df_tmp['End Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    print(df_tmp.loc[:, ['ID', 'Type', 'beginTime', 'endTime']])

# Convert time to seconds for easy computation
def time_to_seconds(t):
    return (t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1e6)

# Setting up file paths
home = expanduser("~")
inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_04142023/*/*.txt")
outFile = home + "/data/ELAN_generated_label/ELAN_generated_label_04142023/label_processed_summary.csv"

# Setting column headers and initializing main dataframe
headers = ["Type", "Begin Time - hh:mm:ss.ms", "Begin Time - ss.msec", "End Time - hh:mm:ss.ms", "End Time - ss.msec",
           "Duration - hh:mm:ss.ms", "Duration - ss.msec", "label", "labelPath", "ID"]
df = pd.DataFrame()

# Parsing input files
for i, inFile in enumerate(inFiles):
    if os.path.getsize(inFile) <= 0: continue

    df_tmp = pd.read_csv(inFile, delimiter='\t', engine='python', header=None)
    if len(df_tmp.columns) != 9:
        print(inFile)
        continue

    df_tmp['labelPath'] = '/'.join((inFile.split('/')[-2:]))
    df_tmp.drop(1, axis=1, inplace=True)
    df_tmp.reset_index(drop=True, inplace=True)

    df_tmp['ID'] = df_tmp['labelPath'].str.extract('(\d+)', expand=False)
    df = pd.concat([df, df_tmp], ignore_index=True)

# Cleaning and formatting dataframe
df.columns = headers
df.drop(columns=['label', 'End Time - ss.msec', 'Begin Time - ss.msec'], inplace=True)
df['Begin Time - hh:mm:ss.ms'] = pd.to_datetime(df['Begin Time - hh:mm:ss.ms'], format='%H:%M:%S.%f')
df['End Time - hh:mm:ss.ms'] = pd.to_datetime(df['End Time - hh:mm:ss.ms'], format='%H:%M:%S.%f')

# Check and associate corresponding .wav files for each label. If no .wav file is found, the label is discarded.
# This is essential because some .wav files might have been moved due to poor quality.
wavPaths = []
nonFileList = []
for sdan in df['ID']:
    wavPath = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/"+str(sdan)+"*.wav"), key=sort_key)
    if len(wavPath) > 1:
        print(wavPath)
        print("Multiple wav files found for a label, check the data.")
    if len(wavPath) == 0:
        nonFileList.append(sdan)
    else:
        wavPaths += wavPath

# Removing data rows with no associated .wav file
for f in nonFileList:
    df = df.drop(df[df['ID'] == f].index)
df['wavPath'] = wavPaths

# Filter the dataset to only include 'Cry [Cr]' and 'Whine/Fuss [F]' types
# df = df[(df['Type'] == 'Cry [Cr]') | (df['Type'] == 'Whine/Fuss [F]')]
print(df['Type'].unique())  # Verifying that the filtering has worked as intended

def time_to_seconds(t):
    return (t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1e6)

def combine_nearby_cry_events(df_sdan):
    """Combine nearby cry events."""
    i = 0
    while i < len(df_sdan) - 1:
        current_row = df_sdan.iloc[i]
        next_row = df_sdan.iloc[i + 1]

        if current_row['Type'] == 'Cry [Cr]' and current_row['Duration - ss.msec'] < 3:
            time_gap = time_to_seconds(next_row['Begin Time - hh:mm:ss.ms'].time()) - time_to_seconds(current_row['End Time - hh:mm:ss.ms'].time())
            if time_gap <= 5:
                # Combine rows
                df_sdan.at[i, 'End Time - hh:mm:ss.ms'] = next_row['End Time - hh:mm:ss.ms']
                df_sdan.at[i, 'Duration - hh:mm:ss.ms'] += next_row['Duration - hh:mm:ss.ms']
                df_sdan.at[i, 'Duration - ss.msec'] += next_row['Duration - ss.msec']
                # Drop the next row
                df_sdan = df_sdan.drop(df_sdan.index[i + 1]).reset_index(drop=True)
            else:
                i += 1
        else:
            i += 1
    return df_sdan

def rename_and_combine_events(df_sdan):
    """Rename labels and combine consecutive events."""
    # df_sdan['Type'] = df_sdan['Type'].replace(['Whine/Fuss [F]', 'Cry [Cr]'], 'Cry')
    df_sdan['Type'] = df_sdan['Type'].replace(['Cry [Cr]'], 'Cry')
    df_sdan['Type'] = df_sdan['Type'].replace(['Whine/Fuss [F]'], 'Whine/Fuss')
    df_sdan['Type'] = df_sdan['Type'].replace(['Scream [S]'], 'Scream')
    df_sdan['Type'] = df_sdan['Type'].replace(['Yell [Y]'], 'Yell')

    df_sdan = df_sdan.sort_values(by='Begin Time - hh:mm:ss.ms', ascending=True).reset_index(drop=True)

    combined_df = pd.DataFrame(columns=df_sdan.columns)
    i = 0
    while i < len(df_sdan) - 1:
        if (df_sdan.loc[i + 1, 'Begin Time - hh:mm:ss.ms'] - df_sdan.loc[
            i, 'End Time - hh:mm:ss.ms']).total_seconds() < 5:
            combined_row = df_sdan.loc[i].copy()
            combined_row['End Time - hh:mm:ss.ms'] = df_sdan.loc[i + 1, 'End Time - hh:mm:ss.ms']
            combined_df = pd.concat([combined_df, combined_row.to_frame().T], ignore_index=True)
            i += 2
        else:
            combined_df = pd.concat([combined_df, df_sdan.iloc[[i]]], ignore_index=True)
            i += 1
    if i == len(df_sdan) - 1:
        combined_df = pd.concat([combined_df, df_sdan.iloc[[i]]], ignore_index=True)
    return combined_df

# Initialize a list to store processed DataFrames
processed_dfs = []

# Iterate over each unique ID
for sdan in df['ID'].unique():
    df_sdan = df[df['ID'] == sdan].reset_index(drop=True)

    # Use the custom function to combine nearby cry events
    df_sdan = combine_nearby_cry_events(df_sdan)

    # Filter out cries shorter than 3 seconds
    df_sdan = df_sdan[~((df_sdan['Type'] == 'Cry [Cr]') & (df_sdan['Duration - ss.msec'] < 3))]

    # Check for any remaining short cries
    short_cries_exist = not df_sdan[(df_sdan['Type'] == 'Cry [Cr]') & (df_sdan['Duration - ss.msec'] < 3)].empty
    if short_cries_exist:
        print('There is an SDAN with a duration less than 3:', sdan)

    # Rename and combine events
    combined_df = rename_and_combine_events(df_sdan)

    # Add the processed DataFrame to the list
    processed_dfs.append(combined_df)

# Concatenate all processed DataFrames
df = pd.concat(processed_dfs, ignore_index=True)

# Save to CSV and print time
df.to_csv(outFile, header=True)
printTime(df)

# Part2: Divide 10 min WAV files into 5x2 min wav files.
wavFolder_2min = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/"

if not os.path.exists(wavFolder_2min):
    os.makedirs(wavFolder_2min)
    print(f"Directory '{wavFolder_2min}' created.")
else:
    print(f"Directory '{wavFolder_2min}' already exists.")

# Generate 2 min wav files from 10 min wav files.
import librosa
import re
import soundfile as sf

wavSet = df['wavPath'].unique()

for index, input_wav in enumerate(wavSet):
    # Extract ID from input wav name
    match = re.search(r'\d+', input_wav.split('/')[-1])
    if not match:
        print(f"No number found in {input_wav}. Skipping...")
        continue
    sdan = match.group(0)

    output_folder = os.path.join(wavFolder_2min, sdan)

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the input WAV file
    try:
        y, sr = librosa.load(input_wav, sr=None)
    except Exception as e:
        print(f"Failed to load {input_wav} due to error: {e}. Skipping...")
        continue

    # Set the duration of each segment to 2 minutes
    segment_duration = 2 * 60
    segment_samples = segment_duration * sr

    # Iterate over the 5 segments
    for i in range(5):
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples
        output_file = os.path.join(output_folder, f'{sdan}_segment_{i}.wav')
        try:
            sf.write(output_file, y[start_sample:end_sample], sr, format='WAV', subtype='PCM_16')
        except Exception as e:
            print(f"Failed to save segment to {output_file} due to error: {e}. Skipping this segment...")

print("Segmentation process completed!")

# Part3: Run Yao's preprocessing/prediction script (refer: previous my code: driver_baseline_concatenate_deBarbaroCry_2min.py)
# goPrediction = False
goPrediction = True
if goPrediction:
    from preprocessing import preprocessing
    from predict import predict

    inFolders = []
    # for sdan in set(IDs):
    # for sdan in df['ID'].unique():
    #     inFolders.append(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan)

    inFolders = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/*"), key=sort_key)

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

        inFiles = sorted(glob.glob(inFolder + '/*.wav'), key=sort_key)
        for inFile in inFiles:
            preprocessedFile = preprocessedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
            predictedFile = predictedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
            probFile = probFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'

            # Run Preproecessing
            print(inFile)
            print(preprocessedFile)
            preprocessing(inFile, preprocessedFile)

            # Run Prediction script
            _, pred_prob = predict(inFile, preprocessedFile, predictedFile, probFile)
            # predict(inFile, preprocessedFile, predictedFile)

# Check the size is 120 (Verification)
inFolders = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/*"), key=sort_key)
for inFolder in inFolders:
    predictedFolder = inFolder + '/predicted/'
    inFiles = sorted(glob.glob(inFolder + '/*.wav'), key=sort_key)
    for inFile in inFiles:
        predictedFile = predictedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
        tmp = pd.read_csv(predictedFile, header=None, names=['Label'])
        if len(tmp) != 120:
            print(len(tmp))


# Concatenate result files (2 min) into 10 min result file.
def concatenate_dataframes(predictionFiles,predictedFolder):
    # Read and store dataframes in a list
    dataframes = []
    for idx, prediction_file in enumerate(predictionFiles):
        df = pd.read_csv(prediction_file, header=None, names=['Label'])

        dataframes.append(df)

    # Concatenate dataframes
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Remove the output file if exists.
    outputFile = predictedFolder + 'concatenated_data.csv'
    if os.path.exists(outputFile):
        os.remove(outputFile)
        print("Removing " + outputFile)

    # Save the concatenated dataframe as a new CSV file
    concatenated_df.to_csv(outputFile, index=False,header=None)

for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/predicted/"
    predictedFiles = glob.glob(predictedFolder + "[0-9].csv")
    # predictedFiles = [file for file in predictedFiles if 'concatenated' not in file and str(sdan)]
    predictedFiles.sort()
    concatenate_dataframes(predictedFiles, predictedFolder)

    probFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/prob/"
    probFiles = glob.glob(predictedFolder + "[0-9].csv")
    # probFiles = [file for file in predictedFiles if 'concatenated' not in file and str(sdan) not in file]
    probFiles.sort()
    concatenate_dataframes(probFiles, probFolder)

# Check the size is 600 (Verification)
for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/predicted/"
    outputFile = predictedFolder + 'concatenated_data.csv'
    tmp = pd.read_csv(outputFile, header=None, names=['Label'])
    if len(tmp) != 600:
        print(len(tmp))


# Part4 Convert labeling result file into second-level ground truth.
import math
for sdan in df['ID'].unique():
    df_sdan = df[df['ID'] == sdan]
    df_sdan.reset_index(drop=True, inplace=True)
    print(sdan)
    df_sdan['beginTime'] = pd.to_timedelta(df_sdan['beginTime'])
    df_sdan['endTime'] = pd.to_timedelta(df_sdan['endTime'])

    # Get the total number of seconds in the dataset
    total_seconds = 600

    # Create a dictionary to store second labels
    second_labels = {}

    for index, row in df_sdan.iterrows():
        begin_sec = math.ceil(row['beginTime'].total_seconds())
        end_sec = math.floor(row['endTime'].total_seconds())
        for sec in range(begin_sec, end_sec+1):
            # second_labels[sec] = 'crying'
            second_labels[sec] = row['Type']
    # Create a new dataframe with second labels
    data = []
    for sec in range(total_seconds):
        if sec in second_labels:
            data.append([sec, sec + 1, second_labels[sec]])
        else:
            # data.append([sec, sec + 1, 'non-crying'])
            data.append([sec, sec + 1, 'None'])

    new_df = pd.DataFrame(data, columns=['Start Time (s)', 'End Time (s)', 'Label'])
    # new_df = pd.DataFrame(data, columns=['Label'])

    # create the output folder if it does not exist
    inFolder = (home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan)
    labelFolder = inFolder + '/groundTruth/'
    labelFile = labelFolder + 'labelFile.csv'
    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)

    # Select only the 'Start Time (s)' and 'Label' columns
    # filtered_df = new_df[['Start Time (s)', 'Label']]
    filtered_df = new_df[['Label']]
    filtered_df['Label'] = filtered_df['Label'].replace({'None': 0, 'Cry': 1, 'Yell': 2, 'Whine/Fuss': 3,'Scream':4})

    # Save the dataframe to a CSV file without headers
    filtered_df.to_csv(labelFile, index=False, header=False)

# Part5: Evaluate the performance of the algorithms.

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
predictionFiles = []
labelFiles = []
probFiles = []
for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/predicted/'
    labelFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/groundTruth/'
    probFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/prob/'

    predictionFiles += sorted(glob.glob(predictedFolder + "concatenated_data.csv"), key=sort_key)
    labelFiles += sorted(glob.glob(labelFolder + "labelFile.csv"), key=sort_key)
    probFiles += sorted(glob.glob(os.path.join(probFolder + "concatenated_data.csv")), key=sort_key)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Initialize a 5x2 confusion matrix
total_confusion = np.zeros((5, 2))

# Dictionaries for label and prediction mappings
prediction_mapping = {0: 'None', 1: 'Crying'}
groundtruth_mapping = {0: 'None', 1: 'Cry', 2: 'Yell', 3: 'Whine/Fuss', 4: 'Scream'}

# Loop through each file pair
for prediction_file, label_file in zip(predictionFiles, labelFiles):
    # Read the files
    pred_df = pd.read_csv(prediction_file, header=None, names=['Prediction'])
    label_df = pd.read_csv(label_file, header=None, names=['Label'])

    for index, label_row in label_df.iterrows():
        label_val = label_row['Label']
        pred_val = pred_df.loc[index, 'Prediction']

        # Skip if label_val is not within our ground truth mapping
        if label_val not in groundtruth_mapping:
            continue

        # Update the corresponding cell in the confusion matrix
        total_confusion[label_val][pred_val] += 1

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
save_path = os.path.join('analysis', 'analysis-10042023', 'normalized_confusion_matrix.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight')
plt.show()





# total_confusion_matrix = None
#
# # Loop through each file pair
# for prediction_file, label_file,prob_df in zip(predictionFiles, labelFiles,probFiles):
#     # Read the files
#     pred_df = pd.read_csv(prediction_file, header=None, names=['Label'])
#     label_df = pd.read_csv(label_file, header=None, names=['Label'])
#
#     # Calculate the confusion matrix for the current file pair
#     current_confusion_matrix = confusion_matrix(label_df['Label'], pred_df['Label'])
#
#     # Add the current confusion matrix to the total confusion matrix
#     if total_confusion_matrix is None:
#         total_confusion_matrix = current_confusion_matrix
#     else:
#         total_confusion_matrix += current_confusion_matrix


# Part6: Analysis

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from os.path import expanduser
import glob
import re
import pandas as pd
import os
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
    plt.savefig('confusion_matrix_lena100.png', dpi=300)
    plt.show()

plot_heatmap(confusion_mat)

# Part7: Analysis II (based on ROC curves)
from sklearn.metrics import roc_curve, auc

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
home = expanduser("~")
predictionFiles = []
labelFiles = []
probFiles = []
for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/predicted/'
    labelFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/groundTruth/'
    probFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/prob/'
    predictionFiles += sorted(glob.glob(predictedFolder + "concatenated_data.csv"), key=sort_key)
    labelFiles += sorted(glob.glob(labelFolder + "labelFile.csv"), key=sort_key)
    probFiles += sorted(glob.glob(os.path.join(probFolder + "concatenated_data.csv")), key=sort_key)

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


# Part8: ROC Curve on one plot
from sklearn.metrics import roc_curve, auc
# 1. Read deBarbaro dataset

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
probs1, labels1 = [], []
for probFile, labelFile in zip(probFiles, labelFiles):
    prob_df = pd.read_csv(probFile, header=None)
    label_df = pd.read_csv(labelFile, header=None)

    probs1.extend(prob_df.values.flatten())
    labels1.extend(label_df.values.flatten())

# 2. READ LENA dataset results.
import os
import re

parent_folder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/"

# Get a list of subfolders
subfolders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]

# Extract numbers from subfolder names
extracted_numbers = []
for subfolder in subfolders:
    # Use regex to find all numbers in the subfolder name
    numbers = re.findall(r'\d+', subfolder)

    # Get the last number (if any) and append it to the extracted_numbers list
    if numbers:
        extracted_numbers.append(int(numbers[-1]))
sdans = subfolders
sdans.sort()


home = expanduser("~")
predictionFiles = []
labelFiles = []
probFiles = []

for sdan in sdans:
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/predicted/'
    labelFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/groundTruth/'
    probFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/prob/'
    predictionFiles += sorted(glob.glob(predictedFolder + "concatenated_data.csv"), key=sort_key)
    labelFiles += sorted(glob.glob(labelFolder + "labelFile.csv"), key=sort_key)
    probFiles += sorted(glob.glob(os.path.join(probFolder + "concatenated_data.csv")), key=sort_key)

# Read data and concatenate them
probs2, labels2 = [], []
for probFile, labelFile in zip(probFiles, labelFiles):
    prob_df = pd.read_csv(probFile, header=None)
    label_df = pd.read_csv(labelFile, header=None)

    probs2.extend(prob_df.values.flatten())
    labels2.extend(label_df.values.flatten())

# Compute ROC curve and ROC area for each group
fpr1, tpr1, _ = roc_curve(labels1, probs1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(labels2, probs2)
roc_auc2 = auc(fpr2, tpr2)

# Plot ROC curve for each group on one figure
plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='darkorange', lw=lw, label='deBarbaro (AUC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='blue', lw=lw, label='LENA100 (AUC = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Precision-recall curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Compute Precision-Recall curve and average precision for each group
precision1, recall1, _ = precision_recall_curve(labels1, probs1)
average_precision1 = average_precision_score(labels1, probs1)

precision2, recall2, _ = precision_recall_curve(labels2, probs2)
average_precision2 = average_precision_score(labels2, probs2)

# Plot Precision-Recall curve for each group on one figure
plt.figure()
lw = 2
plt.plot(recall1, precision1, color='darkorange', lw=lw, label='deBarbaro (AP = %0.2f)' % average_precision1)
plt.plot(recall2, precision2, color='blue', lw=lw, label='LENA100 (AP = %0.2f)' % average_precision2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

