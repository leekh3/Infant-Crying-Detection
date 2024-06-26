# Model: Yao's pre-trained model
# using alex_svm_kyunghun.py, train_alex_kyunghun.py, prediction_kyunghun.py
# training dataset: LENA
# Target: LENA

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

# Getting Subject IDs
import numpy as np
IDs = set()
for inFile in inFiles:
    ID = inFile.split('/')[-2]
    IDs.add(ID)
IDs = list(IDs)
IDs = np.array(IDs)

from sklearn.model_selection import train_test_split
# Split the array into training and testing sets
train_IDs, test_IDs = train_test_split(IDs, test_size=0.3, random_state=29)  # test_size specifies the proportion of the test set

# Setting column headers and initializing main dataframe
headers = ["Type", "Begin Time - hh:mm:ss.ms", "Begin Time - ss.msec", "End Time - hh:mm:ss.ms", "End Time - ss.msec",
           "Duration - hh:mm:ss.ms", "Duration - ss.msec", "label", "labelPath", "ID"]
df = pd.DataFrame()

# Parsing input files
for i, inFile in enumerate(inFiles):
    if os.path.getsize(inFile) <= 0: 
        continue

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

# Part2: Divide dataframe label info into the format used in deBarbaro Training (e.g., alex_svm_kyunghun.py).
# For example, output format should be like:
# 0   400 cry
# 400 405 notcry
# 405 420 cry
# 420 425 notcry
# 425 440 cry
# 440 475 notcry
# 475 480 cry
# 480 485 notcry
# 485 505 cry
# 505 510 notcry
# 510 560 cry
# 560 570 notcry
# 570 580 cry
# 580 585 notcry
# 585 590 cry
# 590 600 notcry

import pandas as pd
import numpy as np
from io import StringIO

# Convert 'beginTime' and 'endTime' to total seconds and apply rounding
df['beginTime'] = pd.to_timedelta(df['beginTime']).dt.total_seconds().apply(np.ceil).astype(int)
df['endTime'] = pd.to_timedelta(df['endTime']).dt.total_seconds().apply(np.floor).astype(int)

# Map 'Type' to 'cry' or 'notcry'
df['Type'] = df['Type'].map(lambda x: 'cry' if x in ['Whine/Fuss', 'Cry'] else 'notcry')

df.drop(columns=['Begin Time - hh:mm:ss.ms', 'End Time - hh:mm:ss.ms',
       'Duration - hh:mm:ss.ms', 'Duration - ss.msec'], inplace=True)


# Function to generate complete timeline for an ID
def generate_timeline(df_id):
    events = []  # Initialize a list to store events
    
    # Initialize the last end time
    last_end_time = 0
    
    for _, row in df_id.iterrows():
        begin_time = row['beginTime']
        end_time = row['endTime']
        status = row['Type']
        
        # If there's a gap between the last end time and the current begin time, fill it with 'notcry'
        if begin_time > last_end_time:
            events.append({'BeginTime': last_end_time, 'EndTime': begin_time, 'Status': 'notcry'})
        
        # Add the current event
        events.append({'BeginTime': begin_time, 'EndTime': end_time, 'Status': status})
        
        # Update the last end time
        last_end_time = end_time
    
    # Ensure the timeline extends to 600 seconds
    if last_end_time < 600:
        events.append({'BeginTime': last_end_time, 'EndTime': 600, 'Status': 'notcry'})
    
    # Convert the list of events to a DataFrame
    timeline = pd.DataFrame(events)
    
    # Convert 'BeginTime' and 'EndTime' to int for consistency
    timeline[['BeginTime', 'EndTime']] = timeline[['BeginTime', 'EndTime']].astype(int)
    
    return timeline

# The base directory (home) needs to be defined. Assuming 'home' is a variable with your base path:
home = os.path.expanduser("~")  # This will get your home directory. Adjust as necessary.

# Specified output folder
outFolder = os.path.join(home, "data/LENA/random_10min_extracted_04142023/kyunghun-10min-data")

# Check if the directory exists, and create it if it does not
if not os.path.exists(outFolder):
    os.makedirs(outFolder)

# Process each ID and save the timeline to CSV
audio_files,annotation_files = [],[]
train_audio_files,train_annotation_files = [],[]
test_audio_files,test_annotation_files = [],[]
IDs = []
for inFile in inFiles:
    id = inFile.split('/')[-2]
    file_path = os.path.join(outFolder, f'{id}.csv')

    if id not in df['ID'].unique():
        wavPath = glob.glob(home + f"/data/LENA/random_10min_extracted_04142023/{id}*.wav")[0]
        timeline_df = {
        'Type': 'notcry',
        # 'labelPath': inFile,
        'labelPath': file_path,
        'ID': ID,
        'wavPath': wavPath,
        'beginTime': 0,  
        'endTime': 600  
        }
        # continue
        timeline_df =  pd.DataFrame([timeline_df])
        # print(inFile)
    else:
        # continue
        timeline_df = df[df['ID']==id]
    timeline_df = generate_timeline(timeline_df)
    timeline_df.to_csv(file_path, index=False,header=False)

    # annotation_files.append(file_path)
    # audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
    annotation_files.append(file_path)
    audio_files.append(wavPath)
    if id in train_IDs:
        # train_audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
        # train_annotation_files.append(file_path)
        train_annotation_files.append(file_path)
        train_audio_files.append(wavPath)
    elif id in test_IDs:
        # test_audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
        # test_annotation_files.append(file_path)
        test_annotation_files.append(file_path)
        test_audio_files.append(wavPath)

def flatten_list(nested_list):
    # This function takes a list, which can contain nested lists, and returns a flat list.
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            # If the element is a list, extend the flat list with the flattened version of this element
            flat_list.extend(flatten_list(element))
        else:
            # If the element is not a list, just append it to the flat list
            flat_list.append(element)
    return flat_list
test_annotation_files = flatten_list(test_annotation_files)
train_annotation_files = flatten_list(train_annotation_files)
train_audio_files = flatten_list(train_audio_files)
test_audio_files = flatten_list(test_audio_files)



# # Process each ID and save the timeline to CSV
# audio_files,annotation_files = [],[]
# for id, df_id in df.groupby('ID'):
#     timeline_df = generate_timeline(df_id)
#     file_path = os.path.join(outFolder, f'{id}.csv')
#     timeline_df.to_csv(file_path, index=False,header=False)
#     annotation_files.append(file_path)
#     audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
    



# for id, df_id in df.groupby('ID'):
#     timeline_df = generate_timeline(df_id)
#     file_path = os.path.join(outFolder, f'{id}.csv')
#     timeline_df.to_csv(file_path, index=False,header=False)


#     annotation_files.append(file_path)
#     audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
#     if id in train_IDs:
#         train_audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
#         train_annotation_files.append(file_path)
#     elif id in test_IDs:
#         test_audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
#         test_annotation_files.append(file_path)

print(len(train_audio_files),len(test_audio_files))

# Part#3: Retrain
alex_model_path = '.trained/train_alex_kyunghun_LENA.h5'
svm_model_path = '.trained/svm_noflip_kyunghun_LENA.joblib'
import sys
sys.path.append('./yao_training')
from train_alex_kyunghun import train_alex
from alex_svm_kyunghun import train_alex_svm

goRetraining = False
if goRetraining:
    # train_alex(audio_files,annotation_files,alex_model_path)
    # train_alex_svm(audio_files,annotation_files,alex_model_path,svm_model_path)
    train_alex(train_audio_files,train_annotation_files,alex_model_path)
    train_alex_svm(train_audio_files,train_annotation_files,alex_model_path,svm_model_path)



# Part4: Prediction
# goPrediction = False
alex_model_path = '.trained/train_alex_kyunghun_LENA.h5'
svm_model_path = '.trained/svm_noflip_kyunghun_LENA.joblib'

goPrediction = True
predicteds = []
from predict_kyunghun import predict_kyunghun
from preprocessing import preprocessing
from prepare_features_for_svm import predict

if goPrediction:
    
    inFolders = []
    # for sdan in set(IDs):
    # for sdan in df['ID'].unique():
    #     inFolders.append(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan)

    inFolders = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/*"), key=sort_key)

    for inFolder in inFolders:
        preprocessedFolder = inFolder + '/preprocessed-LENAtrained/'
        predictedFolder = inFolder + '/predicted-LENAtrained/'
        probFolder = inFolder + '/prob-LENAtrained/'
        # create the output folder if it does not exist
        if not os.path.exists(preprocessedFolder):
            os.makedirs(preprocessedFolder)
        if not os.path.exists(predictedFolder):
            os.makedirs(predictedFolder)
        if not os.path.exists(probFolder):
            os.makedirs(probFolder)

        inFiles = sorted(glob.glob(inFolder + '/*.wav'), key=sort_key)
        for inFile in inFiles:
            id = inFile.split('/')[-2]
            if id in train_IDs:
                continue
            preprocessedFile = preprocessedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
            predictedFile = predictedFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'
            probFile = probFolder + re.findall(r'\d+', inFile.split('/')[-1])[0] + '.csv'

            # Run Preproecessing
            print(inFile)
            print(preprocessedFile)
            preprocessing(inFile, preprocessedFile)

            # Run Prediction script
            # _, pred_prob = predict(inFile, preprocessedFile, predictedFile, probFile,'.trained/svm_kyunghun.joblib')

            audio_filename = inFile
            preprocessed_file = preprocessedFile
            output_file = predictedFile
            prob_file = probFile
            # svm_trained = '.trained/svm_kyunghun.joblib'

            # predicted = predict(inFile, preprocessedFile, predictedFile, probFile, '.trained/svm_kyunghun.joblib')
            # audio_filename,preprocessed_file,output_file,prob_file=None,model1Path=None,model2Path=None
            audio_filename = inFile
            output_file = predictedFile
            model1Path = alex_model_path
            model2Path = svm_model_path
            predicted, _ = predict_kyunghun(inFile, preprocessedFile, predictedFile, probFile, alex_model_path,svm_model_path)
            print(predicted)
            predicteds.append(predicted)
            
import numpy as np
# Flatten the list of arrays
flattened_predicteds = np.concatenate(predicteds)

# Check the size is 120 (Verification)
inFolders = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/*"), key=sort_key)
for inFolder in inFolders:
    predictedFolder = inFolder + '/predicted-LENAtrained/'
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
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/predicted-LENAtrained/"
    predictedFiles = glob.glob(predictedFolder + "[0-9].csv")
    # predictedFiles = [file for file in predictedFiles if 'concatenated' not in file and str(sdan)]
    predictedFiles.sort()
    concatenate_dataframes(predictedFiles, predictedFolder)

    probFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/prob-LENAtrained/"
    probFiles = glob.glob(predictedFolder + "[0-9].csv")
    # probFiles = [file for file in predictedFiles if 'concatenated' not in file and str(sdan) not in file]
    probFiles.sort()
    concatenate_dataframes(probFiles, probFolder)

# Check the size is 600 (Verification)
for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/predicted-LENAtrained/"
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
    # df_sdan['beginTime'] = pd.to_timedelta(df_sdan['beginTime'])
    # df_sdan['endTime'] = pd.to_timedelta(df_sdan['endTime'])
    df_sdan.loc[:, 'beginTime'] = pd.to_timedelta(df_sdan['beginTime'])
    df_sdan.loc[:, 'endTime'] = pd.to_timedelta(df_sdan['endTime'])

    # Get the total number of seconds in the dataset
    total_seconds = 600

    # Create a dictionary to store second labels
    second_labels = {}

    for index, row in df_sdan.iterrows():
        begin_sec = math.ceil(row['beginTime'].total_seconds())
        end_sec = math.floor(row['endTime'].total_seconds())
        for sec in range(begin_sec, end_sec+1):
            # second_labels[sec] = 'crying'
            second_labels[sec] = 'Cry' if row['Type'] == 'cry' else 'None'
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
    labelFolder = inFolder + '/groundTruth-LENAtrained/'
    labelFile = labelFolder + 'labelFile.csv'
    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)

    # Select only the 'Start Time (s)' and 'Label' columns
    # filtered_df = new_df[['Start Time (s)', 'Label']]
    # filtered_df = new_df[['Label']]
    # filtered_df['Label'] = filtered_df['Label'].replace({'None': 0, 'Cry': 1, 'Yell': 2, 'Whine/Fuss': 3,'Scream':4})
    filtered_df = new_df[['Label']].copy()
    filtered_df['Label'] = filtered_df['Label'].replace({'None': 0, 'Cry': 1, 'Yell': 2, 'Whine/Fuss': 3, 'Scream': 4})

    # Save the dataframe to a CSV file without headers
    filtered_df.to_csv(labelFile, index=False, header=False)

# Part5: Evaluate the performance of the algorithms.

# Users/leek13/data/LENA/random_10min_extracted_04052023/segmented_2min/4893/predicted
predictionFiles = []
labelFiles = []
probFiles = []
for sdan in df['ID'].unique():
    predictedFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/predicted-LENAtrained/'
    labelFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/groundTruth-LENAtrained/'
    probFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + '/prob-LENAtrained/'

    predictionFiles += sorted(glob.glob(predictedFolder + "concatenated_data.csv"), key=sort_key)
    labelFiles += sorted(glob.glob(labelFolder + "labelFile.csv"), key=sort_key)
    probFiles += sorted(glob.glob(os.path.join(probFolder + "concatenated_data.csv")), key=sort_key)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Initialize a 2x2 confusion matrix
total_confusion = np.zeros((2, 2))

# Dictionaries for label and prediction mappings
# prediction_mapping = {0: 'None', 1: 'Crying'}
# prediction_mapping = {0: 'None', 1: 'Cry', 2: 'Yell', 3: 'Whine/Fuss', 4: 'Scream'}
# groundtruth_mapping = {0: 'None', 1: 'Cry', 2: 'Yell', 3: 'Whine/Fuss', 4: 'Scream'}


prediction_mapping = {0: 'None', 1: 'Cry'}
groundtruth_mapping = {0: 'None', 1: 'Cry'}

# Loop through each file pair
for prediction_file, label_file in zip(predictionFiles, labelFiles):
    # Read the files
    pred_df = pd.read_csv(prediction_file, header=None, names=['Prediction'])
    label_df = pd.read_csv(label_file,  names=['Label'])

    for index, label_row in label_df.iterrows():
        label_val = label_row['Label']
        pred_val = pred_df.loc[index, 'Prediction']

        # Skip if label_val or pred_val is not within our mappings
        if label_val not in groundtruth_mapping or pred_val not in prediction_mapping:
            continue

        # Update the corresponding cell in the confusion matrix
        total_confusion[label_val][pred_val] += 1


# Normalize the confusion matrix by row
row_sums = total_confusion.sum(axis=1)
normalized_confusion = total_confusion / row_sums[:, np.newaxis]

# Plotting the heatmap
plt.figure(figsize=(12, 9))
ax = sns.heatmap(normalized_confusion, annot=True, cmap="YlGnBu",
                 xticklabels=list(prediction_mapping.values()),
                 yticklabels=list(groundtruth_mapping.values()),
                 fmt=".4%", linewidths=1, linecolor='gray')

# Adding counts to cells
for i, j in itertools.product(range(normalized_confusion.shape[0]), range(normalized_confusion.shape[1])):
    count = total_confusion[i, j]
    plt.text(j + 0.5, i + 0.7, f'\n({int(count)})', ha='center', va='center', color='red', fontsize=12)

# Adjusting labels and title
ax.set_xlabel('Predicted')
ax.set_ylabel('Ground Truth')
ax.set_title('Normalized Confusion Matrix by Row')
plt.yticks(va="center")
plt.tight_layout()

# Saving the figure
save_path = os.path.join('analysis', 'analysis-02042024', 'normalized_confusion_matrix_svm_with_class_weight.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, bbox_inches='tight')
plt.show()

# Direct calculation from the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
true_positives = np.trace(total_confusion)
total_observations = np.sum(total_confusion)
accuracy = true_positives / total_observations

print(f"Accuracy: {accuracy}")