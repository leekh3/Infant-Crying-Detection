import glob
import pandas as pd
import re
import os
import os
import pandas as pd
import glob

def sort_key(file):
    match = re.findall(r'(\d+)', file)
    return int(match[-1]) if match else -1


# Directory for batch files
batch_dir = "batch"
if not os.path.exists(batch_dir):
    os.makedirs(batch_dir)

# Load the Excel file
file_path = 'LENA_IDs_used_for_final_ADAA_analyses.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Assuming the IDs are in the first column, extract the first 100 IDs after the header
sdans = list(df.iloc[:100, 0])

# Get a list of subject folders
inFolders = sorted(glob.glob("acnp/quality/*.csv"), key=sort_key)

# Initialize a list to store results
results = []

# Read Result
for sdan in sdans:
    non_crying_count = 0
    crying_count = 0

    quality_file = f'acnp/quality/{sdan}_quality.csv'
    prediction_file = f'acnp/prediction/{sdan}_prediction.csv'

    # Reading a.csv
    with open(quality_file, 'r') as file:
        quality_list = [int(line.strip()) for line in file]
    # Reading a.csv
    with open(prediction_file, 'r') as file:
        prediction_list = [int(line.strip()) for line in file]

    quality_duration = sum(quality_list)
    crying_count = 0
    non_crying_count = 0
    for i,num in enumerate(prediction_list):
        if quality_list[i] == 1 and prediction_list[i] == 1:
            crying_count += 1
        elif quality_list[i] == 1 and prediction_list[i] == 0:
            non_crying_count += 1

    # Add results to the list
    results.append({"subject": sdan, "non-crying": non_crying_count, "crying": crying_count})

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

# Statistical Analysis - Simple linear regression
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

# Annotating the plot with the p-value of the intercept
plt.text(max(merged_df['crying']) * 0.7, max(merged_df['DBDOS_AM_1LAB']) * 0.9,
         f'Intercept p-value: {model.pvalues[0]:.3f}',
         fontsize=10, color='blue')

# Create directory if it does not exist
output_dir = 'analysis/analysis-11232023acnp'
os.makedirs(output_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(output_dir, 'crying_vs_irritability_annotated.png'))

# Show the plot
plt.show()
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