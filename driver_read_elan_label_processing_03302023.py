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

# Part1: Read generated lables (ELAN)
home = expanduser("~")
inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_04052023/*/*.txt")
outFile = home + "/data/ELAN_generated_label/ELAN_generated_label_04052023/summary.csv"
inFiles.sort()
headers = ["Type","Begin Time - hh:mm:ss.ms","Begin Time - ss.msec","End Time - hh:mm:ss.ms","End Time - ss.msec","Duration - hh:mm:ss.ms",
              "Duration - ss.msec","label","labelPath","SDAN"]
df = pd.DataFrame()
# set display options
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

for i,inFile in enumerate(inFiles):
    if os.path.getsize(inFile) <= 0:
        continue
    # df = pd.read_csv(inFile, header=None)
    df_tmp = pd.read_csv(inFile, delimiter='\t', engine='python',header=None)
    df_tmp['labelPath'] = '/'.join((inFile.split('/')[-2:]))
    df_tmp = df_tmp.drop(1,axis=1)
    df_tmp = df_tmp.reset_index(drop=True)

    df_tmp['SDAN'] = df_tmp['labelPath'].str.extract('(\d+)', expand=False)
    df = pd.concat([df,df_tmp])
df = df.reset_index(drop=True)
df.columns = headers
df = df.drop('label', axis=1)
print(len(df))

# Remove data with no wav file. (there is a label, for wav files can be moved due to bad quality)
wavPaths = []
nonFileList = []
for sdan in df['SDAN']:
    wavPath = glob.glob(home + "/data/LENA/random_10min_extracted_04052023/"+str(sdan)+"*.wav")
    if len(wavPath) == 0:
        nonFileList.append(sdan)
        continue
    else:
        wavPaths += wavPath
for f in nonFileList:
    df = df.drop(df[df['SDAN'] == f].index)
df['wavPath'] = wavPaths

# Filter other rows except for crying and fuss dataset.
df = df[(df['Type'] == 'Cry [Cr]') | (df['Type'] == 'Whine/Fuss [F]')]
print(df['Type'].unique()) # check if type has only Cry and Fuss

for sdan in df['SDAN']:
    df_tmp = df[df['SDAN'] == sdan]











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
for index,input_wav in enumerate(wavSet):
    # Find SDAN from input wav name
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
goPrediction = False
if goPrediction:
    from preprocessing import preprocessing
    from predict import predict

    inFolders = []
    for sdan in set(SDANs):
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


# # Check unique label
# label = df['label']
# label.dropna()
# unique_values = label.unique()
# print(unique_values)
# >>>>>>> c5af09d (doc updated)
