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
for id, df_id in df.groupby('ID'):
    timeline_df = generate_timeline(df_id)
    file_path = os.path.join(outFolder, f'{id}.csv')
    timeline_df.to_csv(file_path, index=False,header=False)
    annotation_files.append(file_path)
    audio_files.append(list(df[df['ID']==id]['wavPath'])[0])
    

# Part#3: Retrain
import sys
sys.path.append('./yao_training')
from train_alex_kyunghun import train_alex
alex_model_path = '.trained/train_alex_kyunghun_LENA.h5'
train_alex(audio_files,annotation_files,alex_model_path)
from alex_svm_kyunghun import train_alex_svm
svm_output_path = '.trained/svm_noflip_kyunghun_LENA.joblib'
train_alex_svm(audio_files,annotation_files,alex_model_path,svm_output_path)
svm_model_path = svm_output_path

# Part4: Prediction
# goPrediction = False
goPrediction = True
predicteds = []
from predict_kyunghun import predict_kyunghun
if goPrediction:
    from preprocessing import preprocessing
    from prepare_features_for_svm import predict

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
            
            predicted = predict_kyunghun(inFile, preprocessedFile, predictedFile, probFile, alex_model_path,svm_model_path)

            predicteds.append(predicted)
            # predict(inFile, preprocessedFile, predictedFile)
import numpy as np
# Flatten the list of arrays
flattened_predicteds = np.concatenate(predicteds)


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
