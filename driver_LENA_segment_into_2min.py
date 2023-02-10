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
inFiles = glob.glob(home + "/data/LENA/*/AN1/*.wav")
inFiles.sort()
outFolders = glob.glob(home + "/data/LENA/*/AN1/")
for i in range(len(outFolders)):
    outFolders[i] = outFolders[i] + 'segmented_2min/'
outFolders.sort()

for i,inFile in enumerate(inFiles):
    # inFile = glob.glob(home + "/data/LENA/1198_LENA/AN1/*.wav")[0]
    # outFolder = home + "/data/LENA/1198_LENA/AN1/segmented_2min/"
    outFolder = outFolders[i]
    def makeDirIfNotExist(path):
    # path = "pythonprog"
    # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(path)
           print("The new directory is created!")
    makeDirIfNotExist(outFolder)

    # Read wav file.
    sound = AudioSegment.from_wav(inFile)
    wavLen = 2 * 60 # 2 min ==> 2*60 sec.

    t1,t2 = 0,wavLen*1000
    fileIdx = 0
    while(t2<len(sound)):
        newAudio = sound[t1:t2]
        newAudio.export(outFolder + '/' + str(fileIdx)+'.wav', format="wav")
        t1 += wavLen*1000
        t2 += wavLen*1000
        fileIdx += 1
    if t1<t2:
        newAudio = sound[t1:]
        newAudio.export(outFolder + '/' + str(fileIdx) + '.wav', format="wav")

