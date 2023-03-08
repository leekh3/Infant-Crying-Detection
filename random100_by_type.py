import pandas as pd
import numpy as np

# specify the path to the input and output CSV files
input_csv_file_path = "w2w_dbdos_anger_modul_score_12012021_Henry_3.3.23.csv"
output_csv_file_path = "random100_from_each_type_kyunghun_3_6_2023.csv"

# read the CSV file into a pandas DataFrame
df = pd.read_csv(input_csv_file_path)

# get a list of unique values in the 'type' column
unique_types = df['type'].unique()

# create an empty DataFrame to store the randomly selected rows
selected_rows = pd.DataFrame(columns=df.columns)

# loop over each unique value in the 'type' column
for t in unique_types:
    # get the rows with the current type value
    type_rows = df[df['type'] == t]
    # randomly select 25 rows from the current type rows
    if len(type_rows) > 25:
        selected_type_rows = type_rows.sample(n=25, random_state=1)
    else:
        selected_type_rows = type_rows
    # append the selected rows to the output DataFrame
    selected_rows = selected_rows.append(selected_type_rows)

# save the selected rows to a new CSV file
selected_rows.to_csv(output_csv_file_path, index=False)

# print a message to confirm the output file was saved
print(f"Randomly selected rows saved to {output_csv_file_path}")
