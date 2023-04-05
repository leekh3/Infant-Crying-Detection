# Thu Mar 30 09:59:45 EDT 2023 by Kyunghun Lee (Created)

#  a script was created to process ELAN-generated labels and produce output
#  suitable for follow-up machine learning analysis using Yao's pre-trained
#  model. The input file consists of LENA labels for 37 infants at 12 months
#  and is 10 minutes in length. The input labels required by the script can be
#  located at "/Volumes/NNT/Lauren Henry Projects/LENA project/Final text files
#  for data analysis" or in the MS Team at "data/FINAL text files for data
#  analysis_03302023.zip".

from os.path import expanduser
import glob

import pandas as pd
import os

# Read the CSV file into a pandas DataFrame object
# df = pd.read_csv('filename.csv', header=None)

# # Iterate through each row in the DataFrame
# for index, row in df.iterrows():
#     # Access the columns in each row using index numbers
#     column1 = row[0]
#     column2 = row[1]
#     # ...and so on for each column
#     print(column1, column2)

home = expanduser("~")
# inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_03302023/*/*.txt")
# inFiles.sort()
# for i,inFile in enumerate(inFiles):
#     if os.path.getsize(inFile) < 0:
#         continue
    # df = pd.read_csv(inFile, header=None)
    # df = pd.read_csv(inFile, delimiter=',| |\t', engine='python',header=None)

inFile = '1752.txt'
header = ["Begin Time - hh:mm:ss.ms","Begin Time - ss.msec","End Time - hh:mm:ss.ms","End Time - ss.msec","Duration - hh:mm:ss.ms",
              "Duration - ss.msec","detection/no-detection"]
df = pd.read_csv(inFile, delimiter='\t', engine='python', header=header)

