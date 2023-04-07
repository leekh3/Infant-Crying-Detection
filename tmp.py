import pandas as pd

# Sample DataFrame
data = {'Type': ['Cry [Cr]', 'Cry [Cr]', 'Laugh [L]', 'Cry [Cr]'],
        'Begin Time - hh:mm:ss.ms': ['00:00:01.000', '00:00:06.000', '00:00:11.000', '00:00:17.000'],
        'End Time - hh:mm:ss.ms': ['00:00:05.000', '00:00:10.000', '00:00:15.000', '00:00:20.000'],
        'Duration - hh:mm:ss.ms': ['00:00:04.000', '00:00:04.000', '00:00:04.000', '00:00:03.000'],
        'Duration - ss.msec': [4.0, 4.0, 4.0, 3.0],
        'labelPath': ['path1', 'path2', 'path3', 'path4'],
        'SDAN': ['SDAN1', 'SDAN2', 'SDAN3', 'SDAN4'],
        'wavPath': ['wav1', 'wav2', 'wav3', 'wav4']}
df = pd.DataFrame(data)
print(df)
print("-------------")
# Convert time columns to datetime objects
time_format = "%H:%M:%S.%f"
df['Begin Time - hh:mm:ss.ms'] = pd.to_datetime(df['Begin Time - hh:mm:ss.ms'], format=time_format)
df['End Time - hh:mm:ss.ms'] = pd.to_datetime(df['End Time - hh:mm:ss.ms'], format=time_format)

# Initialize an empty DataFrame to store the combined rows
combined_df = pd.DataFrame(columns=df.columns)

# Iterate through the DataFrame
i = 0
while i < len(df) - 1:
    if (df.loc[i+1, 'Begin Time - hh:mm:ss.ms'] - df.loc[i, 'End Time - hh:mm:ss.ms']).total_seconds() < 5:
        combined_row = df.loc[i].copy()
        combined_row['End Time - hh:mm:ss.ms'] = df.loc[i+1, 'End Time - hh:mm:ss.ms']
        combined_df = combined_df.append(combined_row)
        i += 2
    else:
        combined_df = combined_df.append(df.loc[i])
        i += 1

# Handle the last row
if i == len(df) - 1:
    combined_df = combined_df.append(df.loc[i])

# Reset index
combined_df.reset_index(drop=True, inplace=True)

# Convert time columns back to string format
combined_df['Begin Time - hh:mm:ss.ms'] = combined_df['Begin Time - hh:mm:ss.ms'].dt.strftime(time_format)
combined_df['End Time - hh:mm:ss.ms'] = combined_df['End Time - hh:mm:ss.ms'].dt.strftime(time_format)

print(combined_df)
