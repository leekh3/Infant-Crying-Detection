import glob
import pandas as pd
import re
import os

def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1

# Define the base path
home = "/Users/leek13/data/LENA/random_10min_extracted_04142023/segmented_2min"

# Get a list of subject folders
inFolders = sorted(glob.glob(home + "/*"), key=sort_key)

# Initialize a list to store results
results = []

# Process each subject folder
for folder in inFolders:
    subject = folder.split('/')[-1]
    non_crying_count = 0
    crying_count = 0

    for i in range(5):
        csv_file = f"{folder}/predicted/{i}.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file, header=None, usecols=[1])
            non_crying_count += (df[1] == 0).sum()
            crying_count += (df[1] != 0).sum()

    # Add results to the list
    results.append({"subject": subject, "non-crying": non_crying_count, "crying": crying_count})

# Create DataFrame from the list
results_df = pd.DataFrame(results)
df = results_df

# Save results to a new CSV file
# results_df.to_csv(home + "/summary_results.csv", index=False)

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read the Excel file
excel_data = pd.read_excel('ADAA_Dataset_4.7.23_FINAL_forKyunghun.xlsx', usecols=['record_id', 'DBDOS_AM_1LAB','papa_in_2'])

# Convert 'record_id' in the Excel data and 'subject' in df to strings
excel_data['record_id'] = excel_data['record_id'].astype(str)
df['subject'] = df['subject'].astype(str)

# Merge with the df DataFrame
merged_df = pd.merge(df, excel_data, left_on='subject', right_on='record_id')

# Statistical Analysis - Simple linear regression as an example
x = merged_df['crying']  # Number of crying events
y = merged_df['DBDOS_AM_1LAB']  # Irritability score
x = sm.add_constant(x)  # adding a constant for linear regression
model = sm.OLS(y, x).fit()

# Visualization
plt.scatter(merged_df['crying'], merged_df['DBDOS_AM_1LAB'])
plt.plot(merged_df['crying'], model.predict(x), color='red')  # regression line
plt.xlabel('Number of Crying Events')
plt.ylabel('Irritability (Anger Modulation) at 12 Months')
plt.title('Relationship between Crying Events and Irritability at 12 Months')
# plt.show()

# Create directory if it does not exist
output_dir = 'analysis/analysis-11232023acnp'
os.makedirs(output_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(output_dir, 'crying_vs_irritability.png'))

# Output the summary of the regression including the p-value
plt.show()
print(model.summary())


import numpy as np

# Remove NaN values from 'y' and the corresponding 'x' values
valid_indices = ~merged_df['papa_in_2'].isnull()  # Indices where 'papa_in_2' is not NaN
x = merged_df.loc[valid_indices, 'crying']  # Only non-NaN crying events
y = merged_df.loc[valid_indices, 'papa_in_2']  # Only non-NaN internalizing symptoms scores

# Add a constant for linear regression and perform OLS
x_with_constant = sm.add_constant(x)  # adding a constant for linear regression
model_papa_in_2 = sm.OLS(y, x_with_constant).fit()

# Visualization for 'papa_in_2'
plt.figure()
plt.scatter(x, y)
plt.plot(x, model_papa_in_2.predict(x_with_constant), color='red')  # regression line
plt.xlabel('Number of Crying Events')
plt.ylabel('Internalizing Symptoms at 24 Months')
plt.title('Relationship between Crying Events and Internalizing Symptoms at 24 Months')

# Create directory if it does not exist for 'papa_in_2'
output_dir_papa_in_2 = 'analysis/analysis-11232023acnp'
os.makedirs(output_dir_papa_in_2, exist_ok=True)

# Save the plot for 'papa_in_2'
plot_path_papa_in_2 = os.path.join(output_dir_papa_in_2, 'crying_vs_internalizing_symptoms.png')
plt.savefig(plot_path_papa_in_2)

# Output the summary of the regression including the p-value for 'papa_in_2'
print(model_papa_in_2.summary())

# Show the plot
plt.show()
#
#
#
#
#
#
#
#
# # Return the path of the saved plot
# saved_plot_path = os.path.join(output_dir, 'crying_vs_irritability.png')
#
#
# # Output the summary of the regression including the p-value
# print(model.summary())