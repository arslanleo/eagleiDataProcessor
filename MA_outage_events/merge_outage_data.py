import pandas as pd
import numpy as np 
import os 
import glob

#Import datasets 
df = pd.read_excel('cleaned_data_state\Raw_state_2023_Massachusetts.xlsx')
print(df.head())
print(df['county'].unique())


folder_path = 'cleaned_data_state'

# Match only the Massachusetts files
file_pattern = os.path.join(folder_path, 'Raw_state_*_Massachusetts.xlsx')
file_list = glob.glob(file_pattern)

print(f"Files to merge: {file_list}")

# Read and stack them
df_list = [pd.read_excel(file) for file in file_list]
merged_df = pd.concat(df_list, ignore_index=True)


merged_df = merged_df.drop(columns=['sum'])
# Save merged file
merged_df.to_csv('merged_massachusetts.csv', index=False)

print(f"Merged {len(file_list)} files, total rows: {len(merged_df)}")
print(merged_df.head())