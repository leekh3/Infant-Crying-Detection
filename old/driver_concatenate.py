import glob
from pathlib import Path
from preprocessing import preprocessing
from predict import predict
import pandas as pd
import wave
from pydub import AudioSegment
import os
# print os.environ['PATH']

# Find input files from input folder.
inFiles = glob.glob("/Users/jimmy/data/P34/*.wav")

labels = []
data = []
soundAll = AudioSegment.from_wav(inFiles[0])
for i in range(1,len(inFiles)):
    print(i)
    if 'notcry' in inFiles[i]:
        labels.append(0)
    else:
        labels.append(1)
    sound2 = AudioSegment.from_wav(inFiles[i])
    soundAll = soundAll + sound2

soundAll.export("output/concatenated_P34.wav", format="wav")
