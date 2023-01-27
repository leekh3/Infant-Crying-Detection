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

# Find input files from input folder.
subFolder = 'P34'
inFolder = "/Users/leek13/data/deBarbaroCry/" + subFolder
inFiles = glob.glob(inFolder + "/*.wav")
# inFiles = glob.glob("/Users/leek13/data/deBarbaroCry/P34/*.wav")
outFolder = "/Users/leek13/data/processed/deBarbaroCry_2min/" + subFolder +'/'

labels = []
data = []
# soundAll = AudioSegment.from_wav(inFiles[0])
soundAll = None

count = 0
fileIdx = 0
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

