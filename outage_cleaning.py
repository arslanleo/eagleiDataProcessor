import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main(state, county, start, end):
    # Set file path and year range
    data_folder = 'Eagle-idatasets/'  # Directory containing raw CSV files
    output_folder = f'./outage_data/{state}/{county}'
    os.makedirs(output_folder, exist_ok=True)
    print("Begun processing the outage data.")
    # state_name = "Illinois"#Select State
    # county_name = "Cook" # Select County
    start=int(start)
    end=int(end)
    # Initialize list to accumulate all years' expanded data
    all_years_data = []

    for year in range(start, end+1):  #change the years as  per your requirements
        try:
            filename = f"eaglei_outages_{year}.csv"
            filepath = os.path.join(data_folder, filename)
            print(f"Processing {filename}...")

            df = pd.read_csv(filepath)

            # Standardize column name
            if 'customers_out' in df.columns:
                df['customers_out'] = df['customers_out']
            elif 'sum' in df.columns:    # For 2023 we have sum columns instead of customers out
                df['customers_out'] = df['sum']
            else:
                print(f"No outage column ('customers_out' or 'sum') in {filename}. Skipping.")
                continue

            # Filter for state and county
            df = df[df['state'] == state]
            df = df[df['county'] == county]

            if df.empty:
                print(f"No data for {county} in {year}. Skipping.")
                continue

            # Forward fill customers_out for missing data
            df['customers_out'] = df['customers_out'].ffill()

            # Convert timestamp
            df['run_start_time'] = pd.to_datetime(df['run_start_time'], errors='coerce')
            df = df.dropna(subset=['run_start_time'])

            # Create complete 15-min interval range
            start_time = df["run_start_time"].min()
            end_time = df["run_start_time"].max()
            full_range = pd.date_range(start=start_time, end=end_time, freq='15min')

            full_df = pd.DataFrame(full_range, columns=['run_start_time'])
            merged_df = pd.merge(full_df, df, on='run_start_time', how='left').ffill() # fill missing timestamps

            # Save cleaned dataset
            #cleaned_path = os.path.join(output_folder, f"Cleaned_data_{year}_{county}.parquet")
            #merged_df.to_parquet(cleaned_path, index=False)

            # ZOH Expansion for sample and hold
            df6 = merged_df.copy()
            df6['sum'] = df6['customers_out']
            df6['run_start_time'] = pd.to_datetime(df6['run_start_time'], errors='coerce')

            expanded_time = []
            expanded_customers_out = []

            for i in range(len(df6) - 1):
                expanded_time.append(df6['run_start_time'].iloc[i])
                expanded_customers_out.append(df6['sum'].iloc[i])
                expanded_time.append(df6['run_start_time'].iloc[i + 1])
                expanded_customers_out.append(df6['sum'].iloc[i])

            expanded_time.append(df6['run_start_time'].iloc[-1])
            expanded_customers_out.append(df6['sum'].iloc[-1])

            df_expanded = pd.DataFrame({
                'run_start_time': expanded_time,
                'sum': expanded_customers_out
            })
            #df_expanded['year'] = year  # Optional: Add year info

            # Save individual expanded dataset
            #zoh_path = os.path.join(output_folder, f"Cleaned_data_{year}_{county}_{state}.parquet")
            #df_expanded.to_parquet(zoh_path, index=False)

            # Append to merged dataset
            all_years_data.append(df_expanded)

            print(f"Year {year} processed and saved.")

        except Exception as e:
            print(f"Error processing {year}: {e}")

    # Merge all expanded datasets
    if all_years_data:
        merged_all = pd.concat(all_years_data, ignore_index=True)
        merged_all.sort_values(by='run_start_time', inplace=True)
        merged_output_path = os.path.join(output_folder, f"Merged_Cleaned_data_{start}_{end}_{county}_{state}.parquet")
        merged_all.to_parquet(merged_output_path, index=False)
        print("All years merged and saved successfully.")
    else:
        print("No data processed to merge.")


# example
# state='Rhode Island'
# county='Bristol'
# start=2018
# end=2019
# main(state,county,start,end)