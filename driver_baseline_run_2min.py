import glob
from pathlib import Path
from preprocessing import preprocessing
from predict import predict
import pandas as pd
import wave
from pydub import AudioSegment
import os
import re
import pandas as pd
# read audio file
from pydub import AudioSegment
# print os.environ['PATH']

def makeDirIfNotExist(path):
# path = "pythonprog"
# Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
       print("The new directory is created!")

# input files
from os.path import expanduser
home = expanduser("~")
inputFolder = "input/2min/"

# # find all folders
# subFolders = glob.glob(inputFolder + 'P*')

# find all folders
inFolders = glob.glob(home + "/data/processed/deBarbaroCry_2min/P*/")

for inFolder in inFolders:

    # subFolder = 'P34'
    # inFolder = home + "/data/processed/deBarbaroCry_2min/" + subFolder + '/'
    subFolder = inFolder.split('/')[-2]

    # preprocessedFolder = inputFolder.replace('input','preprocessed')
    preprocessedFolder = inFolder + '/preprocessed/'
    # outputFolder = inputFolder.replace('input','output')
    outputFolder = inFolder + '/prediction/'
    inputFolder = inFolder
    # Make output folder if it does not exist.
    makeDirIfNotExist(preprocessedFolder)
    makeDirIfNotExist(outputFolder)
    audioFiles = glob.glob(inFolder + "/*.wav")

    for i,audio_filename in enumerate(audioFiles):
        # Determine file names
        fileNumber = int(re.findall(r'\d+', audio_filename.split("/")[-1])[0])
        preprocesedFile = preprocessedFolder + str(fileNumber) + '.csv'
        predictFile = outputFolder + str(fileNumber) + '.csv'
        labelFile = inputFolder + str(fileNumber) + '_label.csv'

        # Preproecessing
        preprocessing(audio_filename, preprocesedFile)

        # Prediction
        predict(audio_filename, preprocesedFile, predictFile)

        # reading the CSV file
        predictData = pd.read_csv(predictFile,header=None, names=['index','prediction'],index_col=0)
        windowData = pd.read_csv(preprocesedFile, header=None, names=['start', 'end'])
        labelData = pd.read_csv(labelFile)

        audio = AudioSegment.from_wav(audio_filename)

        outWavFolder = outputFolder + '/segmented/' + str(fileNumber) + '/'
        makeDirIfNotExist(outWavFolder)
        for i in range(len(predictData)):
            prediction = predictData['prediction'].iloc[i]
            outFile = ''
            if prediction == 0:
                outFile = outWavFolder + str(i) + '_nocry.wav'
            else:
                outFile = outWavFolder + str(i) + '_cry.wav'
            t1 = i*1000
            t2 = (i+1)*1000
            segmentedAudio = audio[t1:t2]
            segmentedAudio.export(outFile, format="wav") #Exports to a wav file in the current path.

