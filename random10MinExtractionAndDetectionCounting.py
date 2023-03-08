import random
from pydub import AudioSegment

import pandas as pd
import numpy as np
# /Volumes/sdan-edb/ELLEN/NNT/Lauren\ Henry\ Projects/LENA\ project/LENA\ Recordings\ \(Restricted\ Access\)/100\ Participants_SelectedforIrritability

def extractRandom10MinFromWav(in_file,out_file):
    # Define constants
    TEN_MIN = 10 * 60 * 1000  # Length of 1 hour segment in milliseconds

    # Load the WAV file
    # wav_file = AudioSegment.from_wav("/Users/leek13/data/LENA/1180/e20171121_094647_013506.wav")
    wav_file = AudioSegment.from_wav(in_file)

    # Calculate the maximum start position for the extracted segment
    max_start_pos = len(wav_file) - TEN_MIN

    # Generate a random start position within this range
    start_pos = random.randint(0, max_start_pos)

    # Calculate the end position for the extracted segment
    end_pos = start_pos + TEN_MIN

    # Extract the 1-hour segment from the WAV file
    extracted_segment = wav_file[start_pos:end_pos]

    # Save the extracted segment to a new WAV file
    extracted_segment.export(out_file, format="wav")

# find folder
# Concatenation (2min)

import glob
from pathlib import Path
from preprocessing import preprocessing
from predict import predict
import pandas as pd
import wave
from pydub import AudioSegment
import os
# print os.environ['PATH']
import csv
from os.path import expanduser

# Find input files from input folder and generate 1 hour random selected wav portion from each file.
# home = expanduser("~")
# inFiles = glob.glob(home + "/data/LENA/*/AN1/*.wav")
dataFolder = '/Volumes/sdan-edb/ELLEN/NNT/Lauren Henry Projects/LENA project/LENA Recordings (Restricted Access)/100 Participants_SelectedforIrritability'
inFiles = glob.glob(dataFolder + '/*/*.wav')
inFiles.sort()
sdan = ''
# for inFile in inFiles:
num_detections = []
sdans = []
file_locations = []
# for i in range(3):
for i in range(len(inFiles)):
    inFile = inFiles[i]
    sdan = inFile.split('/')[-2]
    inFolder = os.path.dirname(inFile)
    # outFolder = inFolder.replace('/LENA/','/LENA_random_10min/')
    outFolder = dataFolder + '/random_10min_extracted/'
    # outFile = inFile.replace('/LENA/','/LENA_random_10min/')
    outFile = outFolder + sdan + '.wav'
    try:
        os.makedirs(outFolder)
        print("folder generated:",outFolder)
    except:
        print("folder already exists:",outFolder)


    # run extractRandom10MinFromWav in inFoiles
    extractRandom10MinFromWav(inFile,outFile)
    print("extracted random 10 min from:",inFile, " To: ",outFile)

# outFolders = glob.glob(home + "/data/LENA/*/AN1/")

# # driver to 2 min
# sound = AudioSegment.from_wav(inFile)
# wavLen = 2 * 60 # 2 min ==> 2*60 sec.
#
# t1,t2 = 0,wavLen*1000
# fileIdx = 0
# print("dividing 1 hour file into multiple 2 min dataset")
# print("input file:",inFile," output Folder:",outFolder)
# while(t2<len(sound)):
#     newAudio = sound[t1:t2]
#     newAudio.export(outFolder + '/' + str(fileIdx)+'.wav', format="wav")
#     t1 += wavLen*1000
#     t2 += wavLen*1000
#     fileIdx += 1
# if t1<t2:
#     newAudio = sound[t1:]
#     newAudio.export(outFolder + '/' + str(fileIdx) + '.wav', format="wav")

# run elan_2min_by_kyunghun.


# Segment input file into 5 sec wav file (will be stored in temprorary folder).
    from segment_into_5sec import segment_into_5sec
    segment_into_5sec(outFile)

    from detection_on_5sec_by_kyunghun import detection_on_5sec_by_kyunghun
    num_detection = detection_on_5sec_by_kyunghun('tmp')

    num_detections.append(num_detection)
    sdans.append(sdan)
    file_locations.append(inFile)


# Save the result.
import pandas as pd

# create 3 example lists
# list1 = ['a', 'b', 'c']
# list2 = [1, 2, 3]
# list3 = ['x', 'y', 'z']

# set the column names explicitly
column_names = ['SDAN', 'file_locations', 'num_detections']

# combine the lists into a pandas DataFrame with specified column names
df = pd.DataFrame(list(zip(sdans, file_locations, num_detections)), columns=column_names)

# save the DataFrame as a CSV file
df.to_csv(outFolder + '/output.csv', index=False)



