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

def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1

def printTime(df_tmp):
    df_tmp['beginTime'] = df_tmp['Begin Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    df_tmp['endTime'] = df_tmp['End Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    print(df_tmp.loc[:, ['ID','Type','beginTime', 'endTime']])

# Part1: Read generated lables (ELAN)
home = expanduser("~")
inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_04142023/*/*.txt")
outFile = home + "/data/ELAN_generated_label/ELAN_generated_label_04142023/label_processed_summary_raw.csv"
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
    wavPath = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/"+str(sdan)+"*.wav"), key=sort_key)
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
# df = df[(df['Type'] == 'Cry [Cr]') | (df['Type'] == 'Whine/Fuss [F]')]
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
    # df_sdan = df_sdan[~((df_sdan['Type'] == 'Cry [Cr]') & (df_sdan['Duration - ss.msec'] < 3))]
    len2 = len(df_sdan)
    # if len1 != len2:
        # print('combined happend' + str(sdan))
    # filtered_df = df_sdan[df_sdan['Type'] == 'Cry [Cr]']
    # all_rows_ge_5 = (filtered_df['Duration - ss.msec'] >= 3).all()
    # if all_rows_ge_5 == False:
    #     print('there is sdan which has duration is less than 3:' + str(sdan))


    # # 2.	Remove rows where the value in the 'Duration - ss.msec' column is less than 3 and corresponds to crying.
    # #  (Fusses did not have a minimum duration)
    # df_sdan = df_sdan.drop(df_sdan[(df_sdan['Duration - ss.msec'] < 3) & (df_sdan['Type'] == 'Cry [Cr]')].index)
    # df_sdan.reset_index(drop=True, inplace=True)
    # printTime(df_sdan)

    # The label names 'Cry' and 'Whine/Fuss [F]' will be changed to 'Cry' since both of them will be treated as crying.
    # df_sdan['Type'] = df_sdan['Type'].replace('Whine/Fuss [F]','Cry')
    df_sdan['Type'] = df_sdan['Type'].replace('Cry [Cr]','Cry')
    df_sdan['Type'] = df_sdan['Type'].replace('Whine/Fuss [F]', 'Whine')
    df_sdan['Type'] = df_sdan['Type'].replace('Yell [Y]', 'Yell')
    df_sdan['Type'] = df_sdan['Type'].replace('Scream [S]', 'Scream')

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

# Part2 Convert labeling result file into second-level ground truth.
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
        print(index,row)
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
            data.append([sec, sec + 1, 'none'])

    new_df = pd.DataFrame(data, columns=['Start Time (s)', 'End Time (s)', 'Label'])
    # new_df = pd.DataFrame(data, columns=['Label'])

    # create the output folder if it does not exist
    inFolder = (home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan)
    labelFolder = inFolder + '/groundTruth_raw/'
    labelFile = labelFolder + 'labelFile.csv'
    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder)

    # Select only the 'Start Time (s)' and 'Label' columns
    # filtered_df = new_df[['Start Time (s)', 'Label']]
    filtered_df = new_df[['Label']]
    # filtered_df['Label'] = filtered_df['Label'].replace({'non-crying': 0, 'crying': 1})

    # Save the dataframe to a CSV file without headers
    filtered_df.to_csv(labelFile, index=False, header=False)

# Part3 Read prediction result and save it as a concatned file.
import math
import pandas as pd
import glob

inFolders = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/*/predicted"), key=sort_key)

for inFolder in inFolders:
    sdan = inFolder.split('/')[-2]
    # Get the list of csv files
    csv_files = sorted(glob.glob(inFolder + "/*.csv"))

    # Initialize an empty dataframe
    df_concat = pd.DataFrame()

    # Loop over the csv files
    for file in csv_files:
        # Read each csv file into a dataframe
        df = pd.read_csv(file, header=None)

        # Concatenate the dataframes
        df_concat = pd.concat([df_concat, df])

    # Reset the index
    df_concat.reset_index(drop=True, inplace=True)

    # Output Folder
    outFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/" + sdan + "/predicted_raw/"
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    # Save the concatenated dataframe as a new csv file
    df_concat.to_csv(outFolder + 'predicted_concatenated.csv', index=False, header=False)

# Part4 Wav file extraction with ground Truth and prediction information.
inFolders = sorted(glob.glob(home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/*/"), key=sort_key)
sdans = []
import pandas as pd
from pydub import AudioSegment

for inFolder in inFolders:
    sdan = inFolder.split('/')[-2]

    # Load ground truth and prediction labels without headers
    ground_truth = pd.read_csv(inFolder + 'groundTruth_raw/labelFile.csv', header=None)
    prediction = pd.read_csv(inFolder + 'predicted_raw/predicted_concatenated.csv', header=None)

    # Get labels. Since there are no headers, use iloc to get the first (and presumably only) column
    ground_truth_labels = ground_truth.iloc[:, 0].values
    prediction_labels = prediction.iloc[:, 1].values
    # / Users / jimmy / data / LENA / random_10min_1sec_result_with_groundtruth_and_prediction

    # For each .wav file
    for i in range(5):
        # Load audio file
        audio = AudioSegment.from_wav(inFolder + f"{i}.wav")

        # For each second
        for j in range(len(audio) // 1000):  # AudioSegment lengths are in milliseconds
            # Extract one second of audio
            one_sec_audio = audio[j * 1000: (j+1) * 1000]

            # Create the folder
            resultFolder = home + '/data/LENA/random_10min_1sec_result_with_groundtruth_and_prediction/' + sdan + '/'
            if not os.path.exists(resultFolder):
                os.makedirs(resultFolder)

            # Create new filename
            new_filename = f"{i*1000+j}_{ground_truth_labels[j]}_{prediction_labels[j]}.wav"

            # Export one second of audio
            one_sec_audio.export(resultFolder + new_filename, format="wav")
