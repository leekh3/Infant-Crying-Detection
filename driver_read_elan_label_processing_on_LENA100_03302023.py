# Thu Mar 30 09:59:45 EDT 2023 by Kyunghun Lee (Created)

#  a script was created to process ELAN-generated labels and produce output
#  suitable for follow-up machine learning analysis using Yao's pre-trained
#  model. The input file consists of LENA labels for 37 infants at 12 months
#  and is 10 minutes in length. The input labels required by the script can be
#  located at "/Volumes/NNT/Lauren Henry Projects/LENA project/Final text files
#  for data analysis" or in the MS Team at "data/FINAL text files for data
#  analysis_03302023.zip".

# Model: Yao's pre-trained model
# Target: LENA dataset. (100 x 10min subject datset)

from os.path import expanduser
import glob
import re
import pandas as pd
import os

def printTime(df_tmp):
    df_tmp['beginTime'] = df_tmp['Begin Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    df_tmp['endTime'] = df_tmp['End Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    print(df_tmp.loc[:, ['ID','Type','beginTime', 'endTime']])

# Part1: Read generated lables (ELAN)
home = expanduser("~")
inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_04142023/*/*.txt")
outFile = home + "/data/ELAN_generated_label/ELAN_generated_label_04142023/label_processed_summary.csv"
inFiles.sort()
headers = ["Type","Begin Time - hh:mm:ss.ms","Begin Time - ss.msec","End Time - hh:mm:ss.ms","End Time - ss.msec","Duration - hh:mm:ss.ms",
              "Duration - ss.msec","label","labelPath","ID"]
df = pd.DataFrame()
# set display options
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

for i,inFile in enumerate(inFiles):
    if os.path.getsize(inFile) <= 0:
        continue
    # df = pd.read_csv(inFile, header=None)
    df_tmp = pd.read_csv(inFile, delimiter='\t', engine='python',header=None)
    if len(df_tmp.columns)!=9:
        print(inFile)
        continue
    print(len(df_tmp.columns))

    df_tmp['labelPath'] = '/'.join((inFile.split('/')[-2:]))
    df_tmp = df_tmp.drop(1,axis=1)
    df_tmp = df_tmp.reset_index(drop=True)

    df_tmp['ID'] = df_tmp['labelPath'].str.extract('(\d+)', expand=False)
    df = pd.concat([df,df_tmp])
df = df.reset_index(drop=True)
df.columns = headers
df = df.drop('label', axis=1)
df = df.drop('End Time - ss.msec', axis=1)
df = df.drop('Begin Time - ss.msec', axis=1)

# Convert the 'Begin Time - hh:mm:ss.ms' and 'End Time - hh:mm:ss.ms' column to datetime format
df['Begin Time - hh:mm:ss.ms'] = pd.to_datetime(df['Begin Time - hh:mm:ss.ms'], format='%H:%M:%S.%f')
df['End Time - hh:mm:ss.ms'] = pd.to_datetime(df['End Time - hh:mm:ss.ms'], format='%H:%M:%S.%f')

# Remove data with no wav file. (there is a label, for wav files can be moved due to bad quality)
wavPaths = []
nonFileList = []
for sdan in df['ID']:
    wavPath = glob.glob(home + "/data/LENA/random_10min_extracted_04142023/"+str(sdan)+"*.wav")
    if len(wavPath)>1:
        print(wavPath)
        print("something wrong")
    if len(wavPath) == 0:
        nonFileList.append(sdan)
        continue
    else:
        wavPaths += wavPath
for f in nonFileList:
    df = df.drop(df[df['ID'] == f].index)
df['wavPath'] = wavPaths

# Filter other rows except for crying and fuss dataset.
df = df[(df['Type'] == 'Cry [Cr]') | (df['Type'] == 'Whine/Fuss [F]')]
print(df['Type'].unique()) # check if type has only Cry and Fuss

df_new = pd.DataFrame()
def time_to_seconds(t):
    return (t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1e6)

for sdan in df['ID'].unique():
    df_sdan = df[df['ID'] == sdan]
    df_sdan.reset_index(drop=True, inplace=True)

    len1 = len(df_sdan)
    i = 0
    while i < len(df_sdan) - 1:
        current_row = df_sdan.iloc[i]
        next_row = df_sdan.iloc[i + 1]

        if current_row['Type'] == 'Cry [Cr]' and current_row['Duration - ss.msec'] < 3:
            time_gap = time_to_seconds(next_row['Begin Time - hh:mm:ss.ms'].time()) - time_to_seconds(current_row['End Time - hh:mm:ss.ms'].time())

            if time_gap <= 5:
                # print('combined:', str(sdan))

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
    #  remove rows with Type equal to 'Cry [Cr]' and a duration less than 3:
    df_sdan = df_sdan[~((df_sdan['Type'] == 'Cry [Cr]') & (df_sdan['Duration - ss.msec'] < 3))]
    len2 = len(df_sdan)
    # if len1 != len2:
        # print('combined happend' + str(sdan))
    filtered_df = df_sdan[df_sdan['Type'] == 'Cry [Cr]']
    all_rows_ge_5 = (filtered_df['Duration - ss.msec'] >= 3).all()
    if all_rows_ge_5 == False:
        print('there is sdan which has duration is less than 3:' + str(sdan))


    # # 2.	Remove rows where the value in the 'Duration - ss.msec' column is less than 3 and corresponds to crying.
    # #  (Fusses did not have a minimum duration)
    # df_sdan = df_sdan.drop(df_sdan[(df_sdan['Duration - ss.msec'] < 3) & (df_sdan['Type'] == 'Cry [Cr]')].index)
    # df_sdan.reset_index(drop=True, inplace=True)
    # printTime(df_sdan)

    # The label names 'Cry' and 'Whine/Fuss [F]' will be changed to 'Cry' since both of them will be treated as crying.
    df_sdan['Type'] = df_sdan['Type'].replace('Whine/Fuss [F]','Cry')
    df_sdan['Type'] = df_sdan['Type'].replace('Cry [Cr]','Cry')
    printTime(df_sdan)

    # Sort the items based on their 'beginTime' value, keeping in mind that they may already be in ascending order
    # and therefore not require any changes.
    df_sdan = df_sdan.sort_values(by='beginTime', ascending=True)
    df_sdan = df_sdan.reset_index(drop=True)
    printTime(df_sdan)

    # Initialize an empty DataFrame to store the combined rows
    combined_df = pd.DataFrame(columns=df_sdan.columns)

    # Iterate through the DataFrame
    i = 0
    while i < len(df_sdan) - 1:
        if (df_sdan.loc[i + 1, 'Begin Time - hh:mm:ss.ms'] - df_sdan.loc[i, 'End Time - hh:mm:ss.ms']).total_seconds() < 5:
            combined_row = df_sdan.loc[i].copy()
            combined_row['End Time - hh:mm:ss.ms'] = df_sdan.loc[i + 1, 'End Time - hh:mm:ss.ms']
            combined_df = combined_df.append(combined_row)
            i += 2
        else:
            combined_df = combined_df.append(df_sdan.loc[i])
            i += 1

    # Handle the last row
    if i == len(df_sdan) - 1:
        combined_df = combined_df.append(df_sdan.loc[i])

    # Reset index
    combined_df.reset_index(drop=True, inplace=True)

    # printTime(combined_df)
    df_new = pd.concat([df_new,combined_df],ignore_index=True)

    # Convert time columns back to string format
    # combined_df['Begin Time - hh:mm:ss.ms'] = combined_df['Begin Time - hh:mm:ss.ms'].dt.strftime("%H:%M:%S.%f")
    # combined_df['End Time - hh:mm:ss.ms'] = combined_df['End Time - hh:mm:ss.ms'].dt.strftime("%H:%M:%S.%f")

df = df_new
df.to_csv(outFile, header=True)
printTime(df)
# Index(['Type', 'Begin Time - hh:mm:ss.ms', 'Begin Time - ss.msec',
#        'End Time - hh:mm:ss.ms', 'End Time - ss.msec',
#        'Duration - hh:mm:ss.ms', 'Duration - ss.msec', 'labelPath', 'ID',
#        'wavPath'],
#       dtype='object')

# Part2: Divide 10 min WAV files into 5x2 min wav files.
wavFolder_2min = home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/"

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
for index,input_wav in enumerate(wavSet):
    # Find ID from input wav name
    match = re.search(r'\d+', input_wav.split('/')[-1])
    sdan = ''
    if match:
        sdan = match.group(0)
    else:
        print(f"No number found.")
        break
    output_folder = wavFolder_2min + sdan

    # create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # load the input WAV file
    y, sr = librosa.load(input_wav, sr=None)

    # set the duration of each segment to 2 minutes
    segment_duration = 2 * 60

    # set the number of samples per segment
    segment_samples = segment_duration * sr

    # iterate over the 5 segments
    for i in range(5):
        # set the start and end samples for the current segment
        start_sample = i * segment_samples
        end_sample = (i + 1) * segment_samples

        # set the output file path
        output_file = os.path.join(output_folder, f'{i}.wav')

        # save the current segment to the output file
        sf.write(output_file, y[start_sample:end_sample], sr, format='WAV', subtype='PCM_16')

# Part3: Run Yao's preprocessing/prediction script (refer: previous my code: driver_baseline_concatenate_deBarbaroCry_2min.py)
# goPrediction = False
goPrediction = False
if goPrediction:
    from preprocessing import preprocessing
    from predict import predict

    inFolders = []
    # for sdan in set(IDs):
    for sdan in df['ID'].unique():
        inFolders.append(home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan)

    for inFolder in inFolders:
        preprocessedFolder = inFolder + '/preprocessed/'
        predictedFolder = inFolder + '/predicted/'
        # create the output folder if it does not exist
        if not os.path.exists(preprocessedFolder):
            os.makedirs(preprocessedFolder)
        if not os.path.exists(predictedFolder):
            os.makedirs(predictedFolder)

        inFiles = glob.glob(inFolder + '/*.wav')
        for inFile in inFiles:
            preprocessedFile = preprocessedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
            predictedFile = predictedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'

            # Run Preproecessing
            print(inFile)
            print(preprocessedFile)
            preprocessing(inFile, preprocessedFile)

            # Run Prediction script
            predict(inFile, preprocessedFile, predictedFile)

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
    predictedFolder = home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan + "/predicted/"
    predictedFiles = glob.glob(predictedFolder + "*.csv")
    predictedFiles = [file for file in predictedFiles if 'concatenated' not in file]
    predictedFiles.sort()
    concatenate_dataframes(predictedFiles, predictedFolder)

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
            second_labels[sec] = 'crying'

    # Create a new dataframe with second labels
    data = []
    for sec in range(total_seconds):
        if sec in second_labels:
            data.append([sec, sec + 1, second_labels[sec]])
        else:
            data.append([sec, sec + 1, 'non-crying'])

    new_df = pd.DataFrame(data, columns=['Start Time (s)', 'End Time (s)', 'Label'])
    # new_df = pd.DataFrame(data, columns=['Label'])

    # create the output folder if it does not exist
    inFolder = (home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan)
    labelFolder = inFolder + '/groundTruth/'
    labelFile = labelFolder + 'labelFile.csv'
    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)

    # Select only the 'Start Time (s)' and 'Label' columns
    # filtered_df = new_df[['Start Time (s)', 'Label']]
    filtered_df = new_df[['Label']]
    filtered_df['Label'] = filtered_df['Label'].replace({'non-crying': 0, 'crying': 1})

    # Save the dataframe to a CSV file without headers
    filtered_df.to_csv(labelFile, index=False, header=False)

# Part5: Evaluate the performance of the algorithms.

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
predictionFiles = []
labelFiles = []
for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan + '/predicted/'
    labelFolder = home + "/data/LENA/random_10min_extracted_04052023/segmented_2min/" + sdan + '/groundTruth/'
    predictionFiles += glob.glob(predictedFolder + "concatenated_data.csv")
    labelFiles += glob.glob(labelFolder + "labelFile.csv")


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


# Part6: Analysis

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