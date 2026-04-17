# to get the dataset of the beijing health care

import pandas as pd

# Load the dataset from CSV
df = pd.read_csv('heart_attack_china.csv')  # replace with your filename

# Filter rows where the 'Region' column indicates Beijing
beijing_df = df[df['Province'].str.contains('Beijing', case=False, na=False)]

# Save the filtered dataset to a new CSV file
beijing_df.to_csv('beijing_data.csv', index=False)