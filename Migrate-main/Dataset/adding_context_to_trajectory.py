import pandas as pd

# Load datasets
taxi_df = pd.read_csv('combined_taxi_data.csv')
health_df = pd.read_csv('beijing_data.csv')

# Get unique taxi_id count
unique_taxi_ids = taxi_df['taxi_id'].drop_duplicates()
num_unique = len(unique_taxi_ids)

# Determine sample size (minimum of 1777 or total unique taxi IDs)
sample_size = min(1777, num_unique)

# Sample taxi_id(s)
sampled_taxi_ids = unique_taxi_ids.sample(n=sample_size, random_state=42)

# Sample equal number of health records
sampled_health = health_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# Pair sampled taxi_ids with sampled health info
health_sampled = sampled_health.copy()
health_sampled['taxi_id'] = sampled_taxi_ids.values

# Select only needed columns
health_sampled = health_sampled[['taxi_id', 'Age', 'Physical_Activity', 'CVD_Risk_Score', 'Hypertension', 'Diabetes']]

# Merge with entire taxi dataset
merged_df = taxi_df.merge(health_sampled, on='taxi_id', how='left')

# Save the output
merged_df.to_csv('taxi_with_health_info.csv', index=False)