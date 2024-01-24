# Reliabiltiy evaluation of different results from two coders.
# Input file: Kyunghun 1.4.24

# 1. Read each CSV file: Load the CSV files listed in inFiles and inFiles2.
# 2. Filter rows: Keep only the rows where the first column is "Cry [Cr]" or "Whine/Fuss [F]".
# 3. Time Processing: Round down the start time and round up the end time.
# 4. Create 600-rows file: For each second in the 10-minute interval (600 seconds), label it as "Cry" if it falls within any of the time intervals from step 3, otherwise label it as "Not Cry".
# 5. Comparison for Accuracy: Compare the generated files from inFiles with those from inFiles2 on a second-by-second basis to calculate the accuracy.
# 6. Calculate Overall Accuracy: Calculate the overall accuracy across all files.

import pandas as pd
import glob

def time_to_seconds(time_str):
    """Convert time string 'HH:MM:SS.sss' to seconds."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def process_file(file_path):
    try:
        # Read CSV file
        df = pd.read_csv(file_path, delimiter='\t', header=None)

        # Check if the file is empty
        if df.empty:
            return ['Not Cry'] * 600

        # Filter rows
        df = df[df[0].isin(['Cry [Cr]', 'Whine/Fuss [F]'])]

        # Process times and create 600 seconds data
        seconds_data = ['Not Cry'] * 600
        for _, row in df.iterrows():
            start_time = int(time_to_seconds(row[2]))  # Convert and round down start time
            end_time = int(time_to_seconds(row[4]) + 0.999)  # Convert and round up end time
            for i in range(start_time, min(end_time, 600)):
                seconds_data[i] = 'Cry'

        return seconds_data
    except pd.errors.EmptyDataError:
        # Return 'Not Cry' for all 600 seconds if the file is empty
        return ['Not Cry'] * 600

def compare_files(file1, file2):
    correct = sum(x == y for x, y in zip(file1, file2))
    return correct / 600

# Example usage
inFiles = glob.glob('/Users/leek13/data/LENA/Kyunghun 1.4.24/*/*.txt')
inFiles1, inFiles2 = [], []

for i, inFile in enumerate(inFiles):
    if i % 2 == 0:
        inFiles1.append(inFile)
    else:
        inFiles2.append(inFile)

# Process each file
processed_files1 = [process_file(f) for f in inFiles1]
processed_files2 = [process_file(f) for f in inFiles2]

# Calculate accuracy for each pair of files
accuracies = [compare_files(f1, f2) for f1, f2 in zip(processed_files1, processed_files2)]

results = []
def find_coder(inFile):
    coder_list = ['LH', 'EH', 'TE']
    coder = 'others'
    for coder_candidate in coder_list:
        if coder_candidate in inFile:
            coder = coder_candidate
    return coder

for i in range(len(inFiles1)):
    inFile1,inFile2 = inFiles1[i],inFiles2[i]
    sdan = inFile1.split('/')[-2]
    accuracy = accuracies[i]
    coder1 = find_coder(inFile1)
    coder2 = find_coder(inFile2)
    results.append([str(sdan),str(accuracy),coder1,coder2])

# Calculate overall accuracy
overall_accuracy = sum(accuracies) / len(accuracies)

print("Accuracies per file:", accuracies)
print("Overall Accuracy:", overall_accuracy)

import os
analysis_dir = 'analysis/analysis-01232024'
os.makedirs(analysis_dir, exist_ok=True)

# Save results to CSV
results_df = pd.DataFrame(results, columns=['filename', 'accuracy', 'coder1', 'coder2'])
results_df.to_csv(os.path.join(analysis_dir, 'accuracy_analysis.csv'), index=False)

print("Results saved to 'accuracy_analysis.csv'")