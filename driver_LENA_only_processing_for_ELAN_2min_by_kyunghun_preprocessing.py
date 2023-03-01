import glob
from pathlib import Path
from preprocessing_kyunghun import preprocessing_kyunghun
from predict import predict
import pandas as pd
import wave
from pydub import AudioSegment
import os
import re
import pandas as pd
# read audio file
from pydub import AudioSegment
from time import strftime
from time import gmtime
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

inFolders = glob.glob(home + "/data/LENA/*/AN1/segmented_2min/")
subjects = glob.glob(home + "/data/LENA/*/")
for i in range(len(subjects)):
    subjects[i] = subjects[i].split('/')[-2]
inFolders.sort()
subjects.sort()

# inFolder = glob.glob(home + "/data/LENA/1198_LENA/AN1/segmented_2min/")[0]
for i in range(len(inFolders)):
    inFolder = inFolders[i]
    subject = subjects[i]


    # for inFolder in inFolders:

    # subFolder = 'P34'
    # inFolder = home + "/data/processed/deBarbaroCry_2min/" + subFolder + '/'
    # subFolder = inFolder.split('/')[-2]

    # preprocessedFolder = inputFolder.replace('input','preprocessed')
    preprocessedFolder = inFolder + '/preprocessed_kyunghun/'
    # outputFolder = inputFolder.replace('input','output')
    # outputFolder = inFolder + '/prediction/'
    inputFolder = inFolder
    # Make output folder if it does not exist.
    makeDirIfNotExist(preprocessedFolder)
    # makeDirIfNotExist(outputFolder)
    audioFiles = glob.glob(inFolder + "/*.wav")

    header = ["Begin Time - hh:mm:ss.ms","Begin Time - ss.msec","End Time - hh:mm:ss.ms","End Time - ss.msec","Duration - hh:mm:ss.ms",
              "Duration - ss.msec","detection/no-detection"]

    pOuts = []
    # for i,audio_filename in enumerate(audioFiles):
    pOutsBackup = pOuts
    for j in range(len(audioFiles)):
        audio_filename = inFolder + str(j) + '.wav'
        # Determine file names
        fileNumber = int(re.findall(r'\d+', audio_filename.split("/")[-1])[0])
        preprocesedFile = preprocessedFolder + str(fileNumber) + '.csv'
        labelFile = inputFolder + str(fileNumber) + '_label.csv'

        # Preproecessing
        pOut = preprocessing_kyunghun(audio_filename, preprocesedFile)
        pOuts.append(pOut)

    pOutsBackup = pOuts

    for j in range(len(pOuts)):
        for k in range(len(pOuts[j])):
            pOuts[j][k][0]+=i*120
            pOuts[j][k][1]+=i*120

    pOutsSerial = []
    for j in range(len(pOuts)):
        for k in range(len(pOuts[j])):
            pOutsSerial.append(pOuts[j][k])

    startIdx = pOutsSerial[0][0]
    endIdx = pOutsSerial[0][1]
    pOutsFinal = []
    for j in range(1,len(pOutsSerial)):
        if endIdx>=pOutsSerial[j][0]:
            endIdx = pOutsSerial[j][1]
        else:
            pOutsFinal.append([startIdx,endIdx])
            startIdx = pOutsSerial[j][0]
            endIdx = pOutsSerial[j][1]
    pOutsFinal.append([startIdx,endIdx])

    # Chcek if everyhting is good
    for j in range(len(pOutsFinal)-1):
        if pOutsFinal[j][1]>=pOutsFinal[j+1][0]:
            print("bad")

    pOuts = pOutsFinal
    # nOuts = []
    # if pOuts[0][0] != 0:
    #     nOuts.append([0,pOuts[0][0]])
    # for i in range(len(pOuts)-1):
    #     nOuts.append([pOuts[i][1],pOuts[i+1][0]])

    l0,l1,l2,l3,l4,l5,l6 = [],[],[],[],[],[],[]
    for j in range(len(pOuts)):
        start = pOuts[j][0]
        end = pOuts[j][1]
        beginTime = strftime("%H:%M:%S", gmtime(start)) + '.000'
        beginTime2 = str(start) + '.00'

        endTime = strftime("%H:%M:%S", gmtime(end)) + '.000'
        endTime2 = str(end) + '.00'

        duration = strftime("%H:%M:%S", gmtime(end)) + '.000'
        duration2 = str(end) + '.00'

        l0.append(beginTime)
        l1.append(beginTime2)
        l2.append(endTime)
        l3.append(endTime2)
        l4.append(duration)
        l5.append(duration2)
        l6.append("detection")

    df = pd.DataFrame()
    df[header[0]] = l0
    df[header[1]] = l1
    df[header[2]] = l2
    df[header[3]] = l3
    df[header[4]] = l4
    df[header[5]] = l5
    df[header[6]] = l6

    # df.to_csv('ELAN_2min_detection.csv', sep='\t')
    df.to_csv('output_detection_for_elan/'+ subject + '_ELAN_2min_detection.csv', sep='\t')

