# Thu Mar 30 09:59:45 EDT 2023 by Kyunghun Lee (Created)

from os.path import expanduser
import glob
import re
import pandas as pd
import csv
import os
from pydub import AudioSegment

# Concatenation (2min)
home = expanduser("~")
# find all folders
inFolders = glob.glob(home + "/data/deBarbaroCry/P*")
inFolders.sort()

def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1

for inFolder in inFolders:
    labelFile = inFolder + '/label.csv'
    subFolder = inFolder.split('/')[-1]

    # Sort the files by their number
    wav_files = sorted(glob.glob(os.path.join(inFolder, subFolder + '_*.wav')), key=sort_key)

    # Set the maximum duration for each combined file (2 minutes in this case)
    max_duration = 2 * 60 * 1000  # milliseconds

    combined_audio = AudioSegment.empty()
    file_counter = 0

    labels = []
    for wav_file in wav_files:
        # audio = AudioSegment.from_wav(wav_file)
        # combined_audio += audio

        # Add label based on 'cry' or 'notcry' in file name
        if 'notcry' in wav_file:
            labels.extend([0])
        elif 'cry' in wav_file:
            labels.extend([1])
    # Open the file in 'write' mode and write the list to it
    with open(labelFile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the list to the CSV file, with each element as a separate row
        for item in labels:
            writer.writerow([item])



# ra_annotations: [[0, 1.0, 'other'], [1.0, 243.0, 'fuss']...]


