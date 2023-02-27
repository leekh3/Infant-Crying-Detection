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
home = expanduser("~")
# Find input files from input folder.

# find all folders
inFolders = glob.glob(home + "/data/deBarbaroCry/P*/")

for inFolder in inFolders:

# subFolder = 'P34'
#     inFolder = home + "/data/deBarbaroCry/" + subFolder
    # inFolder = "/home/"
    subFolder = inFolder.split('/')[-2]
    inFiles = glob.glob(inFolder + "/*.wav")
    # inFiles = glob.glob("/Users/leek13/data/deBarbaroCry/P34/*.wav")
    outFolder = home + "/data/processed/deBarbaroCry_2min/" + subFolder +'/'

    labels = []
    data = []
    # soundAll = AudioSegment.from_wav(inFiles[0])
    soundAll = None

    count = 0
    fileIdx = 0
    # Make output folder if it does not exist.
    def makeDirIfNotExist(path):
    # path = "pythonprog"
    # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")


    makeDirIfNotExist(outFolder)
    for i in range(0,len(inFiles)):
        count += 1
        print(count)

        if 'notcry' in inFiles[i]:
            labels.append(0)
        else:
            labels.append(1)
        if soundAll != None:
            sound2 = AudioSegment.from_wav(inFiles[i])
            soundAll = soundAll + sound2
        else:
            soundAll = AudioSegment.from_wav(inFiles[i])

        if count == 24:
            soundAll.export(outFolder + str(fileIdx) +".wav", format="wav")
            myFile = outFolder+ str(fileIdx) + "_label.csv"
            df = pd.DataFrame(data={"label": labels})
            df.to_csv(myFile, sep=',', index=False)
            count = 0
            fileIdx += 1
            labels = []
            soundAll = None

    if soundAll !=None:
        soundAll.export(outFolder + str(fileIdx) + ".wav", format="wav")
        myFile = outFolder + str(fileIdx) + "_label.csv"
        df = pd.DataFrame(data={"label": labels})
        df.to_csv(myFile, sep=',', index=False)
        count = 0
        fileIdx += 1
        labels = []
        soundAll = None

