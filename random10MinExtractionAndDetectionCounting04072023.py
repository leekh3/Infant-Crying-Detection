import random
from pydub import AudioSegment
import glob
from pydub import AudioSegment
import os
import pandas as pd
import numpy as np
# /Volumes/sdan-edb/ELLEN/NNT/Lauren\ Henry\ Projects/LENA\ project/LENA\ Recordings\ \(Restricted\ Access\)/100\ Participants_SelectedforIrritability
def extractRandom10MinFromWav(in_file,out_file,start_pos,end_pos=-1):
    # Define constants
    TEN_MIN = 10 * 60 * 1000  # Length of 1 hour segment in milliseconds

    # Load the WAV file
    # wav_file = AudioSegment.from_wav("/Users/leek13/data/LENA/1180/e20171121_094647_013506.wav")
    wav_file = AudioSegment.from_wav(in_file)

    # Calculate the maximum start position for the extracted segment
    max_start_pos = 0
    if end_pos == -1:
        max_start_pos = len(wav_file) - TEN_MIN
    else:
        max_start_pos = end_pos - TEN_MIN

    if max_start_pos - start_pos < TEN_MIN:
        print("cannot extract 10 min from this start_pos,end_pos.")
        return

    # Generate a random start position within this range
    start_pos = random.randint(0, max_start_pos)

    # Calculate the end position for the extracted segment
    end_pos = start_pos + TEN_MIN

    # Extract the 1-hour segment from the WAV file
    extracted_segment = wav_file[start_pos:end_pos]

    # Save the extracted segment to a new WAV file
    extracted_segment.export(out_file, format="wav")
    return start_pos,end_pos

def MinSecToMilliseconds(hour,min,sec):
    return (3600*hour+(min*60)+sec)*1000

def calculateNumOfDetect(outFile):
    from segment_into_5sec import segment_into_5sec
    segment_into_5sec(outFile)

    from detection_on_5sec_by_kyunghun import detection_on_5sec_by_kyunghun
    num_detection = detection_on_5sec_by_kyunghun('tmp')

    # num_detections.append(num_detection)

    return num_detection

# Find input files from input folder and generate 1 hour random selected wav portion from each file.
# dataFolder = '/Volumes/sdan-edb/ELLEN/NNT/Lauren Henry Projects/LENA project/LENA Recordings (Restricted Access)/100 Participants_SelectedforIrritability_USE ME/'
dataFolder = '/Users/leek13/Desktop/data_tmp/'
inFiles = glob.glob(dataFolder + '/*/*.wav')
inFiles.sort()
sdan = ''
# for inFile in inFiles:
num_detections = []
sdans = []
file_locations = []
start_points = []
# for i in range(3):

inFiles_new = []
start_points = []
end_points = []
for inFile in inFiles:
    if 'old' in inFile:
        continue
    elif 'random' in inFile:
        continue
    elif '/3999/' in inFile:
        inFiles_new.append(inFile)
        start_points.append(MinSecToMilliseconds(0,0,0))
        end_points.append(MinSecToMilliseconds(8, 0, 0))
    # elif '/2021/' in inFile:
    #     inFiles_new.append(inFile)
    #     start_points.append(MinSecToMilliseconds(1,6,0))
    #     end_points.append(MinSecToMilliseconds(1, 30, 0))
    # elif '/4319/' in inFile:
    #     inFiles_new.append(inFile)
    #     start_points.append(MinSecToMilliseconds(0,0,0))
    #     end_points.append(MinSecToMilliseconds(2,51,0))
    # elif '/2816/' in inFile:
    #     inFiles_new.append(inFile)
    #     start_points.append(-0)
    #     end_points.append(-1)
inFiles = inFiles_new

S = []
E = []
for i in range(len(inFiles)):
    inFile = inFiles[i]

    sdan = inFile.split('/')[-2]
    inFolder = os.path.dirname(inFile)
    if inFolder.split('/')[-1].isnumeric() == False:
        print("skipped:", inFile)
        continue
    outFolder = dataFolder + '/random_10min_extracted/'
    outFile = outFolder + sdan + '_updated_by_kyunghun_04062023.wav'
    try:
        os.makedirs(outFolder)
        print("folder generated:",outFolder)
    except:
        print("folder already exists:",outFolder)

    # run extractRandom10MinFromWav in inFoiles
    for j in range(10):
        print("extracted random 10 min from:", inFile, " To: ", outFile)
        s,e = extractRandom10MinFromWav(inFile, outFile,start_points[i],end_points[i])
        numOfDetect = calculateNumOfDetect(outFile)
        if numOfDetect > 10:
            print("numOfDetect:",",good!,",outFile)
            break
        elif j < 9:
            print("numOfDetect:", ",trying again,",outFile)
        else:
            print("numOfDetect:", ",but # of trials reached 10, so we keep this copy:",outFile)

    sdans.append(sdan)
    num_detections.append(numOfDetect)
    S.append(s)
    E.append(e)
    file_locations.append(inFile)

# set the column names explicitly
column_names = ['SDAN', 'file_locations', 'num_detections','start_point','end_point']

# combine the lists into a pandas DataFrame with specified column names
df = pd.DataFrame(list(zip(sdans, file_locations, num_detections,S,E)), columns=column_names)

# save the DataFrame as a CSV file
df.to_csv(outFolder + '/output_summary_04062023.csv', index=False)