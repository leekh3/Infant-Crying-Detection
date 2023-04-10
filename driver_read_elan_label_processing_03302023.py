# Thu Mar 30 09:59:45 EDT 2023 by Kyunghun Lee (Created)

#  a script was created to process ELAN-generated labels and produce output
#  suitable for follow-up machine learning analysis using Yao's pre-trained
#  model. The input file consists of LENA labels for 37 infants at 12 months
#  and is 10 minutes in length. The input labels required by the script can be
#  located at "/Volumes/NNT/Lauren Henry Projects/LENA project/Final text files
#  for data analysis" or in the MS Team at "data/FINAL text files for data
#  analysis_03302023.zip".

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
inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_04102023/*/*.txt")
outFile = home + "/data/ELAN_generated_label/ELAN_generated_label_04102023/label_processed_summary.csv"
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
    wavPath = glob.glob(home + "/data/LENA/random_10min_extracted_04102023/"+str(sdan)+"*.wav")
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
for sdan in df['ID'].unique():
    df_sdan = df[df['ID'] == sdan]
    df_sdan.reset_index(drop=True, inplace=True)
    print(sdan)
    printTime(df_sdan)
    print('-----')

    # 2.	Remove rows where the value in the 'Duration - ss.msec' column is less than 3 and corresponds to crying.
    #  (Fusses did not have a minimum duration)
    df_sdan = df_sdan.drop(df_sdan[(df_sdan['Duration - ss.msec'] < 3) & (df_sdan['Type'] == 'Cry [Cr]')].index)
    df_sdan.reset_index(drop=True, inplace=True)
    printTime(df_sdan)

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

    printTime(combined_df)
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
goPrediction = True
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

# Part4 Convert labeling result file into second-level ground truth.
for sdan in df['ID'].unique():
    df_sdan = df[df['ID'] == sdan]
    df_sdan.reset_index(drop=True, inplace=True)
    print(sdan)


