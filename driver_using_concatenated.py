import glob
from pathlib import Path
from preprocessing import preprocessing
from predict import predict
import pandas as pd
import wave
from pydub import AudioSegment
import os
# print os.environ['PATH']

audio_filename = 'output/concatenated_P34.wav'
preprocessed_file = 'preprocessed/preprocessed.csv'
predict_file = 'output/predicted.csv'
# preprocessing(audio_filename,preprocessed_file)
# predict(audio_filename,preprocessed_file,output_file)

# Analysis
inFiles = glob.glob("/Users/jimmy/data/P34/*.wav" )
labels = []
for i in range(1,len(inFiles)):
    # print(i)
    if 'notcry' in inFiles[i]:
        labels.append(0)
    else:
        labels.append(1)

import csv
import pandas

# reading the CSV file
predictData = pandas.read_csv(predict_file,header=None, names=['index','prediction'],index_col=0)
windowData = pandas.read_csv(preprocessed_file,header=None, names=['start','end'])

# read audio file
from pydub import AudioSegment
# t1 = 0
# t2 = 1
# t1 = t1 * 1000 #Works in milliseconds
# t2 = t2 * 1000
# audio = AudioSegment.from_wav(audio_filename)
# newAudio = audio[t1:t2]
# newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.

audio = AudioSegment.from_wav(audio_filename)
for i in range(len(predictData)):
    label = predictData['prediction'].iloc[i]
    outFile = ''
    if label == 0:
        outFile = 'output/segmented/' + str(i) + '_nocry.wav'
    else:
        outFile = 'output/segmented/' + str(i) + '_cry.wav'
    t1 = i*1000
    t2 = (i+1)*1000
    segmentedAudio = audio[t1:t2]
    segmentedAudio.export(outFile, format="wav") #Exports to a wav file in the current path.

