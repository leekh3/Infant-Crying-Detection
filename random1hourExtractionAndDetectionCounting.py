import random
from pydub import AudioSegment

def extractRandom1hourFromWav(in_file,out_file):
    # Define constants
    ONE_HOUR = 60 * 60 * 1000  # Length of 1 hour segment in milliseconds

    # Load the WAV file
    # wav_file = AudioSegment.from_wav("/Users/leek13/data/LENA/1180/e20171121_094647_013506.wav")
    wav_file = AudioSegment.from_wav(in_file)

    # Calculate the maximum start position for the extracted segment
    max_start_pos = len(wav_file) - ONE_HOUR

    # Generate a random start position within this range
    start_pos = random.randint(0, max_start_pos)

    # Calculate the end position for the extracted segment
    end_pos = start_pos + ONE_HOUR

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
home = expanduser("~")
inFiles = glob.glob(home + "/data/LENA/*/AN1/*.wav")
inFiles.sort()
for inFile in inFiles:
    inFolder = os.path.dirname(inFile)
    outFolder = inFolder.replace('/LENA/','/LENA_random_1hour/')
    outFile = inFile.replace('/LENA/','/LENA_random_1hour/')
    try:
        os.makedirs(outFolder)
        print("folder generated:",outFolder)
    except:
        print("folder already exists:",outFolder)


    # run extractRandom1hourFromWav in inFoiles
    extractRandom1hourFromWav(inFile,outFile)
    print("extracted random 1 hour from:",inFile, " To: ",outFile)

outFolders = glob.glob(home + "/data/LENA/*/AN1/")

# driver to 2 min
sound = AudioSegment.from_wav(inFile)
wavLen = 2 * 60 # 2 min ==> 2*60 sec.

t1,t2 = 0,wavLen*1000
fileIdx = 0
print("dividing 1 hour file into multiple 2 min dataset")
print("input file:",inFile," output Folder:",outFolder)
while(t2<len(sound)):
    newAudio = sound[t1:t2]
    newAudio.export(outFolder + '/' + str(fileIdx)+'.wav', format="wav")
    t1 += wavLen*1000
    t2 += wavLen*1000
    fileIdx += 1
if t1<t2:
    newAudio = sound[t1:]
    newAudio.export(outFolder + '/' + str(fileIdx) + '.wav', format="wav")

# run elan_2min_by_kyunghun.
