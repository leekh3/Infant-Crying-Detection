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
inFiles = glob.glob("/Users/jimmy/data/P34/*.wav")

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
        soundAll.export("input/2min/" + str(fileIdx) +".wav", format="wav")
        myFile = "input/2min/" + str(fileIdx) + "_label.csv"
        df = pd.DataFrame(data={"label": labels})
        df.to_csv(myFile, sep=',', index=False)
        count = 0
        fileIdx += 1
        labels = []
        soundAll = None

if soundAll !=None:
    soundAll.export("input/2min/" + str(fileIdx) + ".wav", format="wav")
    myFile = "input/2min/" + str(fileIdx) + "_label.csv"
    df = pd.DataFrame(data={"label": labels})
    df.to_csv(myFile, sep=',', index=False)
    count = 0
    fileIdx += 1
    labels = []
    soundAll = None

