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
import re
import pandas as pd
import os

def printTime(df_tmp):
    df_tmp['beginTime'] = df_tmp['Begin Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    df_tmp['endTime'] = df_tmp['End Time - hh:mm:ss.ms'].dt.strftime('%H:%M:%S.%f')
    print(df_tmp.loc[:, ['ID','Type','beginTime', 'endTime']])

# Part1: Read generated lables (ELAN)
home = expanduser("~")
inFiles = glob.glob(home + "/data/ELAN_generated_label/ELAN_generated_label_04102023/*/*.txt")
outFile = home + "/data/ELAN_generated_label/ELAN_generated_label_04102023/label_processed_summary.csv"
inFiles.sort()
headers = ["Type","Begin Time - hh:mm:ss.ms","Begin Time - ss.msec","End Time - hh:mm:ss.ms","End Time - ss.msec","Duration - hh:mm:ss.ms",
              "Duration - ss.msec","label","labelPath","ID"]
df = pd.DataFrame()
# set display options
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

for i,inFile in enumerate(inFiles):
    if os.path.getsize(inFile) <= 0:
        continue
    # df = pd.read_csv(inFile, header=None)
    df_tmp = pd.read_csv(inFile, delimiter='\t', engine='python',header=None)
    if len(df_tmp.columns)!=9:
        print(inFile)
        continue
    print(len(df_tmp.columns))

    df_tmp['labelPath'] = '/'.join((inFile.split('/')[-2:]))
    df_tmp = df_tmp.drop(1,axis=1)
    df_tmp = df_tmp.reset_index(drop=True)

    df_tmp['ID'] = df_tmp['labelPath'].str.extract('(\d+)', expand=False)
    df = pd.concat([df,df_tmp])
df = df.reset_index(drop=True)
df.columns = headers
df = df.drop('label', axis=1)
df = df.drop('End Time - ss.msec', axis=1)
df = df.drop('Begin Time - ss.msec', axis=1)

# Convert the 'Begin Time - hh:mm:ss.ms' and 'End Time - hh:mm:ss.ms' column to datetime format
df['Begin Time - hh:mm:ss.ms'] = pd.to_datetime(df['Begin Time - hh:mm:ss.ms'], format='%H:%M:%S.%f')
df['End Time - hh:mm:ss.ms'] = pd.to_datetime(df['End Time - hh:mm:ss.ms'], format='%H:%M:%S.%f')

# Remove data with no wav file. (there is a label, for wav files can be moved due to bad quality)
wavPaths = []
nonFileList = []
for sdan in df['ID']:
    wavPath = glob.glob(home + "/data/LENA/random_10min_extracted_04102023/"+str(sdan)+"*.wav")
    if len(wavPath)>1:
        print(wavPath)
        print("something wrong")
    if len(wavPath) == 0:
        nonFileList.append(sdan)
        continue
    else:
        wavPaths += wavPath
for f in nonFileList:
    df = df.drop(df[df['ID'] == f].index)
df['wavPath'] = wavPaths

# Total duration for each event type:
total_duration_by_type = df.groupby('Type')['Duration - ss.msec'].sum()
print(total_duration_by_type)

# Average duration for each event type:
avg_duration_by_type = df.groupby('Type')['Duration - ss.msec'].mean()
print(avg_duration_by_type)


# Pie chart1
import matplotlib.pyplot as plt
# Pie chart data
sizes = [80, 20]
labels = ['Has Data', 'No Data']
colors = ['#4CAF50', '#F44336']
explode = (0.1, 0)  # explode the 'Has Data' slice

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Data Availability')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
# plt.show()

# Save the pie chart as a PNG image
plt.savefig('pie_chart1.png', dpi=300, bbox_inches='tight')

# Bar chart (average duration)
import matplotlib.pyplot as plt

# Data for the bar plot
import matplotlib.pyplot as plt

# Data for the bar plot
event_types = ['Cry [Cr]', 'Scream [S]', 'Whine/Fuss [F]', 'Yell [Y]']
avg_durations = [1.124318, 1.502539, 0.770015, 1.150174]
colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']

# Plot the bar plot
plt.figure(figsize=(8, 6))
plt.bar(event_types, avg_durations, color=colors)
plt.title('Average Duration by Event Type')
plt.xlabel('Event Type')
plt.ylabel('Duration (ss.msec)')

# Save the bar plot as a PNG image
plt.savefig('bar_plot.png', dpi=300, bbox_inches='tight')


# Pie chart #2
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with a 'Type' column
event_counts = df['Type'].value_counts()

# Plot the pie chart with different colors for each event type
colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(event_counts, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Event Count by Type')
plt.axis('equal')

# Create a legend with event type labels
legend_labels = [f'{etype}: {count}' for etype, count in event_counts.items()]
plt.legend(wedges, legend_labels, loc='lower right', bbox_to_anchor=(1, 0))

# Save the pie chart as a PNG image
# plt.savefig('event_count_pie_chart_percentage_only.png', dpi=300, bbox_inches='tight')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Data for the pie chart
event_types = ['Cry [Cr]', 'Scream [S]', 'Whine/Fuss [F]', 'Yell [Y]']
total_durations = [746.547, 114.193, 1212.003, 132.270]

# Plot the pie chart
colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
plt.figure(figsize=(8, 6))
plt.pie(total_durations, labels=event_types, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Total Duration by Event Type (sec)')
plt.axis('equal')

# Save the pie chart as a PNG image
# plt.savefig('total_duration_pie_chart3.png', dpi=300, bbox_inches='tight')

# Show the pie chart (optional, since we are saving it as a file)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Data for the pie chart
event_types = ['Cry [Cr]', 'Scream [S]', 'Whine/Fuss [F]', 'Yell [Y]']
total_durations = [746.547, 114.193, 1212.003, 132.270]

# Plot the pie chart
colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
plt.figure(figsize=(8, 6))
# wedges, texts, autotexts = plt.pie(total_durations, labels=event_types, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
wedges, texts, autotexts = plt.pie(total_durations, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Total Duration by Event Type')
plt.axis('equal')

# Write the total duration number on the bottom right of the window
total_duration = sum(total_durations)
plt.text(1, -1, f'Total Duration: {total_duration:.3f}', fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes)

# Save the pie chart as a PNG image
plt.savefig('total_duration_pie_chart_with_number.png', dpi=300, bbox_inches='tight')

# Show the pie chart (optional, since we are saving it as a file)
plt.show()


# Pie chart #2
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with a 'Type' column
event_counts =  [746.547, 114.193, 1212.003, 132.270]

# Plot the pie chart with different colors for each event type
colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
plt.figure(figsize=(8, 6))
wedges, texts, autotexts = plt.pie(event_counts, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Total Duration by Event Type')
plt.axis('equal')

# Create a legend with event type labels
legend_labels = [f'{etype}: {count}' for etype, count in event_counts.items()]
plt.legend(wedges, legend_labels, loc='lower right', bbox_to_anchor=(1, 0))

# Save the pie chart as a PNG image
# plt.savefig('event_count_pie_chart_percentage_only.png', dpi=300, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt

event_types = ['Cry [Cr]', 'Scream [S]', 'Whine/Fuss [F]', 'Yell [Y]']
total_durations = [746.547, 114.193, 1212.003, 132.270]

colors = ['#FFC107', '#FF5722', '#4CAF50', '#2196F3']  # define colors for each slice

fig, ax = plt.subplots()

ax.pie(total_durations, colors=colors, labels=event_types, autopct='%1.1f%%',
       startangle=90, counterclock=False)

ax.legend(loc='lower right')  # add a legend to the chart

plt.show()