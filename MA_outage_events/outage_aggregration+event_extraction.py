import pandas as pd

# Load your dataset
df = pd.read_csv('merged_massachusetts_with_coords.csv')

# Make sure run_start_time is datetime
df['run_start_time'] = pd.to_datetime(df['run_start_time'])

# Sort by county and time
df = df.sort_values(['county', 'run_start_time']).reset_index(drop=True)

# Create outage groups where 'customers_out' changes within each county
def assign_groups(x):
    return (x != x.shift()).cumsum()

df['outage'] = df.groupby('county')['customers_out'].transform(assign_groups)

# Aggregate: get first & last timestamp for each constant-outage block
agg_df = (
    df.groupby(['county', 'outage', 'customers_out', 'latitude', 'longitude'], as_index=False)
      .agg(start_time=('run_start_time', 'first'),
           end_time=('run_start_time', 'last'))
)

# Calculate outage duration in hours
agg_df['outage_duration_hours'] = (agg_df['end_time'] - agg_df['start_time']).dt.total_seconds() / 3600

# Replace zero durations with 0.25 hours
agg_df['outage_duration_hours'] = agg_df['outage_duration_hours'].replace(0, 0.25)


# Save the result
agg_df.to_csv('county_outage_blocks.csv', index=False)

print("Outage block summary saved as 'county_outage_blocks.csv'")
print(agg_df.head(10))
