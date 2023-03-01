import pandas as pd

# create 3 example lists
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
list3 = ['x', 'y', 'z']

# set the column names explicitly
column_names = ['Column A', 'Column B', 'Column C']

# combine the lists into a pandas DataFrame with specified column names
df = pd.DataFrame(list(zip(list1, list2, list3)), columns=column_names)

# save the DataFrame as a CSV file
df.to_csv('output.csv', index=False)