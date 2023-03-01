# Concatenation (2min)

# def segment_into_2min(inFile,outFolder):

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

inFile = inFiles[0]
inFolder = os.path.dirname(inFile)
outFolder = inFolder.replace('/LENA/','/LENA_random_1hour/')
try:
    os.makedirs(outFolder)
    print("folder generated:", outFolder)
except:
    print("folder already exists:", outFolder)
