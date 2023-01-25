import glob
from pathlib import Path
from preprocessing import preprocessing
from predict import predict
import pandas as pd

# Find input files from input folder.
inFiles = glob.glob("input/*.wav")
print(inFiles)

# Determine preprocessing,output file name.
preFiles = []
outFiles = []
for inFile in inFiles:
    preFile = inFile.replace('.wav','_preprocessed.csv')
    preFile = preFile.replace('input','preprocessed')
    outFile = inFile.replace('.wav','.csv')
    outFile = outFile.replace('input','output')
    preFiles.append(preFile)
    outFiles.append(outFile)
print(preFiles)
print(outFiles)

# Run 'preprocessing' script and display the result.
for i in range(len(inFiles)):
    inFile,preFile = inFiles[i],preFiles[i]
    preprocessing(inFile,preFile)

# Run 'predict' script and display the result.
for i in range(len(inFiles)):
    inFile,preFile,outFile = inFiles[i],preFiles[i],outFiles[i]
    predict(inFile,preFile,outFile)