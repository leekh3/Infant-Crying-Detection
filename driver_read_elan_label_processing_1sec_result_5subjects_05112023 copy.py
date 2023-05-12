# SCRIPT FOR THIS REQUEST:
#
# LENA 2021: 1:31 – 2:26
# LENA 2161: 2:12 – 3:06
# LENA 2379: 1:11 – 1: 23; 5:01 – 5:14
# LENA 2520: 0:00 – 00:14; 1:19 – 1:35
# LENA 2643: 4:07 – 4:23

# Model: Yao's pre-trained model
# Target: LENA dataset. (100 x 10min subject datset)

from os.path import expanduser
import glob
import re
import pandas as pd
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import glob
home = expanduser("~")
baseFolder = home + "/data/LENA/random_10min_extracted_04142023/segmented_2min/"
outFolder = home + "/Downloads/result_5subjects"
if not os.path.exists(outFolder):
    os.makedirs(outFolder)
import pandas as pd
from collections import Counter

# Define subjects and their intervals
subjects = {
    "2021": [("1:31", "2:26")],
    "2161": [("2:12", "3:06")],
    "2379": [("1:11", "1:23"), ("5:01", "5:14")],
    "2520": [("0:00", "00:14"), ("1:19", "1:35")],
    "2643": [("4:07", "4:23")]
}
# Define all possible combinations of ground truth and predictions
ground_truth_labels = ['none', 'Whine', 'Cry']
prediction_labels = ['no-Cry', 'Cry']
all_combinations = [(gt, pred) for gt in ground_truth_labels for pred in prediction_labels]

for subject, intervals in subjects.items():
    df_ground_truth = pd.read_csv(f"{baseFolder}/{subject}/groundTruth_raw/labelFile.csv", header=None)
    df_predicted = pd.read_csv(f"{baseFolder}/{subject}/predicted_raw/predicted_concatenated.csv", header=None, usecols=[1])

    df_predicted[1] = df_predicted[1].map({0: 'no-Cry', 1: 'Cry'})

    for i, interval in enumerate(intervals):
        start, end = map(lambda t: int(t.split(':')[0]) * 60 + int(t.split(':')[1]), interval)

        # Initialize confusion matrix for this subject and interval
        confusion_matrix = pd.DataFrame(0, index=ground_truth_labels, columns=prediction_labels)

        for t in range(start, end):
            timestamp_start = f"{t // 60}:{t % 60:02}"
            timestamp_end = f"{(t + 1) // 60}:{(t + 1) % 60:02}"

            ground_truth = df_ground_truth.iloc[t, 0]
            prediction = df_predicted.iloc[t, 0]

            df = pd.DataFrame({"timestamp (start)": [timestamp_start], "timestamp (end)": [timestamp_end], "ground truth": [ground_truth], "prediction": [prediction]})
            df.to_csv(f"{outFolder}/{subject}_{i}.csv", mode='a', index=False)

            # Update the confusion matrix
            confusion_matrix.at[ground_truth, prediction] += 1

        # Calculate the row percentages
        row_sum = confusion_matrix.sum(axis=1)
        percentage_matrix = confusion_matrix.div(row_sum, axis=0)

        # Save the percentage matrix as a CSV file
        percentage_matrix.to_csv(f"{outFolder}/{subject}_{i}_percentage_confusion_matrix.csv")

        # Create a heatmap
        plt.figure(figsize=(10,7))
        sns.heatmap(percentage_matrix, annot=True, cmap='YlGnBu')
        plt.title(f'Percentage Confusion Matrix for {subject}_{i}')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')

        # Save the figure
        plt.savefig(f"{outFolder}/{subject}_{i}_percentage_confusion_matrix.png")

        # Display the figure
        plt.show()







from pydub import AudioSegment
# Extracting the relevant audio segments:
for subject, intervals in subjects.items():
    audio = AudioSegment.from_wav(f"{baseFolder}/{subject}/0.wav") + \
             AudioSegment.from_wav(f"{baseFolder}/{subject}/1.wav") + \
             AudioSegment.from_wav(f"{baseFolder}/{subject}/2.wav") + \
             AudioSegment.from_wav(f"{baseFolder}/{subject}/3.wav") + \
             AudioSegment.from_wav(f"{baseFolder}/{subject}/4.wav")

    for i, interval in enumerate(intervals):
        start, end = map(lambda t: int(t.split(':')[0]) * 60 * 1000 + int(t.split(':')[1]) * 1000, interval)  # convert to milliseconds
        audio_segment = audio[start:end]
        audio_segment.export(f"{outFolder}/{subject}_{i}.wav", format="wav")