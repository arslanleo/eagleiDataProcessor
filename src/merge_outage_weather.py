import pandas as pd
import numpy as np
import os
import xarray as xr

from src.configManager import ConfigManager

def merge_outages_with_weather_netcdf(state: str, 
                                      county: str, 
                                      start: int, 
                                      end: int, 
                                      config: ConfigManager) -> None:
    """
    Merge outage data with weather NetCDF file, adding outage variables as time-series data.
    
    This function loads an existing weather NetCDF file (with station-based weather data)
    and adds county-level outage data as additional time-series variables. The resulting
    merged NetCDF allows users to select any time range and access both weather data
    (from any station) and outage data for that period.
    
    Parameters:
    -----------
    state : str
        State name
    county : str
        County name
    start : int
        Start year
    end : int
        End year
    config : ConfigManager
        Configuration manager object
    """
    
    print(f"Merging outage data with weather NetCDF for {county}, {state} ({start}-{end})")
    
    # Create file paths
    weather_file_dir = os.path.join(config.get("data_paths.weather_data_dir"), state)
    weather_file_name = config.get("file_patterns.cleaned_weather_file_pattern").format(state=state, county=county, start=start, end=end)
    weather_file_path = os.path.join(weather_file_dir, weather_file_name)

    outage_file_dir = os.path.join(config.get("data_paths.outage_data_dir"), state)
    outage_file_name = config.get("file_patterns.cleaned_outage_file_pattern").format(start=start, end=end, county=county, state=state)
    outage_file_path = os.path.join(outage_file_dir, outage_file_name)
    
    # Output path for merged NetCDF
    merged_file_dir = os.path.join(config.get("data_paths.merged_data_dir"), state)
    merged_file_name = config.get("file_patterns.merged_file_pattern").format(start=start, end=end, county=county, state=state)
    merged_file_path = os.path.join(merged_file_dir, merged_file_name)
    
    # Check if weather NetCDF exists
    if not os.path.exists(weather_file_path):
        raise FileNotFoundError(f"Weather NetCDF file not found: {weather_file_path}")
    
    if not os.path.exists(outage_file_path):
        raise FileNotFoundError(f"Outage data file not found: {outage_file_path}")
    
    # Load weather NetCDF dataset
    print(f"Loading weather NetCDF from {weather_file_path}")
    ds_weather = xr.open_dataset(weather_file_path)
    
    # Load outage data
    print(f"Loading outage data from {outage_file_path}")
    df_outage = pd.read_parquet(outage_file_path)
    
    # Get the time coordinate from weather dataset
    time_coord = ds_weather.coords['time']
    
    # Create a dataframe with full time range
    full_time_df = pd.DataFrame({'time': pd.to_datetime(time_coord.values)})

    # find the name of the event number column in outage data, that starts with 'event_number_'
    event_number_col = [col for col in df_outage.columns if col.startswith('event_number_')]
    if len(event_number_col) > 0:
        event_number_col = event_number_col[0]
    else:
        event_number_col = None
        raise ValueError("No event number column found in outage data.")
    
    # Merge outage data with full time range
    df_outage_full = full_time_df.merge(
        df_outage[['run_start_time', 'customers_out', event_number_col]].rename(columns={'run_start_time': 'time'}),
        on='time',
        how='left'
    )
    
    # Fill missing outage values with 0 (no outage)
    df_outage_full['customers_out'] = df_outage_full['customers_out'].fillna(0)
    df_outage_full[event_number_col] = df_outage_full[event_number_col].fillna(0)
    
    # Convert to numpy array
    outage_customers = df_outage_full['customers_out'].values.astype(np.int64)
    outage_event_numbers = df_outage_full[event_number_col].values.astype(np.int64)
    
    # Add outage data as new variables to the dataset
    ds_weather['customers_out'] = (['time'], outage_customers)
    ds_weather['customers_out'].attrs = {
        'long_name': 'Number of customers without power',
        'units': 'count',
        'description': 'Total number of customers experiencing power outage'
    }
    ds_weather['event_number_eaglei'] = (['time'], outage_event_numbers)
    ds_weather['event_number_eaglei'].attrs = {
        'long_name': 'Outage event number',
        'units': 'count',
        'description': f'Identifier for distinct outage events, based on a customer threshold of {event_number_col.split("_")[-1]}'
    }
    
    # # Calculate additional outage metrics if MCC data is available
    # try:
    #     mcc_file_path = os.path.join(config.get("data_paths.eaglei_data_dir"), 'MCC.csv')
    #     county_fips_file_path = os.path.join(config.get("data_paths.eaglei_data_dir"), 'county_fips_master.csv')
        
    #     if os.path.exists(mcc_file_path) and os.path.exists(county_fips_file_path):
    #         # Load MCC Data
    #         pdf = pd.read_csv(mcc_file_path)
    #         county_to_fips = pd.read_csv(county_fips_file_path, encoding='latin')
            
    #         # Find total number of customers in county
    #         ans = county_to_fips[county_to_fips['county_name'] == f'{county} County']
    #         ans = ans[ans['state_name'] == state]
            
    #         if len(ans) > 0:
    #             target_fips = ans['fips'].values[0]
    #             pdf['County_FIPS'] = pd.to_numeric(pdf['County_FIPS'], downcast='integer', errors='coerce')
    #             result = pdf[pdf['County_FIPS'] == target_fips]
                
    #             if len(result) > 0:
    #                 total_county_customers = result['Customers'].values[0]
                    
    #                 # Calculate normalized outage (fraction of total customers)
    #                 normalized_outage = outage_customers / total_county_customers
    #                 ds_weather['customers_out_normalized'] = (['time'], normalized_outage.astype(np.float32))
    #                 ds_weather['customers_out_normalized'].attrs = {
    #                     'long_name': 'Normalized customer outages',
    #                     'units': 'fraction',
    #                     'description': f'Fraction of total customers affected (total: {total_county_customers})'
    #                 }
                    
    #                 # Add total customers as an attribute
    #                 ds_weather.attrs['total_county_customers'] = int(total_county_customers)
                    
    #                 print(f"Added normalized outage data (total customers: {total_county_customers})")
    # except Exception as e:
    #     print(f"Note: Could not calculate normalized outage metrics: {e}")
    
    # Update dataset attributes
    ds_weather.attrs.update({
        'title': f'Merged Weather and Outage Data for {county}, {state} ({start}-{end})',
        'description': 'This dataset contains weather data from multiple stations within the county, along with county-level outage data as time-series variables.',
        'state': state,
        'county': county,
        'county_fips_code': str(df_outage['fips_code'].unique()[0]) if 'fips_code' in df_outage.columns else 'unknown',
        'start_year': start,
        'end_year': end,
        'temporal_resolution': '15 minutes',
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    # Remove an existing attribute if present
    if 'min_data_threshold' in ds_weather.attrs:
        del ds_weather.attrs['min_data_threshold']

    # Find total customers in the county and add as attribute
    customers_data_file_path = os.path.join(
        config.get("data_paths.outage_data_dir"),
        state,
        f"county_total_customers_in_{state.lower()}.parquet"
    )
    # check if the file exists
    if os.path.exists(customers_data_file_path):
        customers_df = pd.read_parquet(customers_data_file_path)
        total_customers = customers_df[customers_df['county'] == county]['total_customers'].values
        if len(total_customers) > 0:
            ds_weather.attrs['total_county_customers'] = int(total_customers[0])
        else:
            ds_weather.attrs['total_county_customers'] = 'unknown'
    else:
        ds_weather.attrs['total_county_customers'] = 'unknown'
        print(f"Note: Total customers data file not found: {customers_data_file_path}")
    
    # Save merged dataset
    print(f"Saving merged NetCDF to {merged_file_path}")
    # create directory if it does not exist
    os.makedirs(merged_file_dir, exist_ok=True)
    ds_weather.to_netcdf(merged_file_path)
    
    # Close the dataset
    ds_weather.close()
    
    print(f"Successfully merged outage and weather data!")
    print(f"  - Time points: {len(time_coord)}")
    print(f"  - Weather stations: {len(ds_weather.coords['station'])}")
    print(f"  - Weather variables: {[v for v in ds_weather.data_vars if v not in ['customers_out', 'customers_out_normalized', 'event_number_eaglei']]}")
    print(f"  - Outage variables: {['customers_out', 'event_number_eaglei']}")
    
    return None


# def main(state: str, 
#          county: str, 
#          start: int, 
#          end: int, 
#          config: ConfigManager) -> None:
    
#     print("Aligning outage and weather data.")

#     # create file paths
#     mcc_file_path = os.path.join(config.get("data_paths.eaglei_data_dir"), 'MCC.csv')
#     county_fips_file_path = os.path.join(config.get("data_paths.eaglei_data_dir"), 'county_fips_master.csv')

#     weather_file_dir = os.path.join(config.get("data_paths.weather_data_dir"), state)
#     weather_file_name = config.get("file_patterns.cleaned_weather_file_pattern").format(state=state, county=county, start=start, end=end)
#     weather_file_path = os.path.join(weather_file_dir, weather_file_name)

#     outage_file_dir = os.path.join(config.get("data_paths.outage_data_dir"), state)
#     outage_file_name = config.get("file_patterns.cleaned_outage_file_pattern").format(start=start, end=end, county=county, state=state)
#     outage_file_path = os.path.join(outage_file_dir, outage_file_name)
    
#     # Load MCC Data
#     pdf = pd.read_csv(mcc_file_path)

#     # find total number of customers in county
#     county_to_fips = pd.read_csv(county_fips_file_path, encoding='latin')
#     ans = county_to_fips[county_to_fips['county_name']==f'{county} County']
#     ans = ans[ans['state_name']==state]
#     target_fips = ans['fips'].values[0]
#     pdf['County_FIPS'] = pd.to_numeric(pdf['County_FIPS'], downcast='integer',errors='coerce')
#     result = pdf[pdf['County_FIPS'] == target_fips]
#     total_county_customers = result['Customers'].values[0]

#     threshold=round(config.get("data_cleaning_parameters.outage_threshold_percentage")*int(total_county_customers))
#     print(total_county_customers, threshold)

#     # Load DataSets; Outage and Weather DataSets
#     df_outage_data = pd.read_parquet(outage_file_path)
#     df_weather_data = pd.read_parquet(weather_file_path)


#     # Convert 'valid' column to datetime in weather data
#     df_weather_data['DATE'] = pd.to_datetime(df_weather_data['valid'])

#     # drop first row in outage data that weather does not interpolate
#     df_outage_data=df_outage_data[(df_outage_data['run_start_time']!=f'{start}-01-01 00:00:00')]
    
#     # Filter data for the specified date range (start to end years, inclusive)
#     df_outage_data = df_outage_data[(df_outage_data['run_start_time'] >= f'{start}-01-01') & (df_outage_data['run_start_time'] < f'{end+1}-01-01')]
#     df_weather_data = df_weather_data[(df_weather_data['DATE'] >= f'{start}-01-01') & (df_weather_data['DATE'] < f'{end+1}-01-01')]

#     # Set datetime as index for outage and weather data
#     df_outage_data.set_index('run_start_time', inplace = True, drop=False)
#     df_weather_data.set_index('DATE', inplace=True, drop=False)

#     # Convert all the weather variables to float
#     weather_vars = config.get("weather_variables").values()
#     for var in weather_vars:
#         df_weather_data[var] = pd.to_numeric(df_weather_data[var], errors='coerce').astype(float)

#     df_outage_data['sum'] = df_outage_data['sum'].apply(lambda x: 0 if x < threshold else x)  # Apply threshold
#     df_outage_data['N_sum'] = df_outage_data['sum'] / total_county_customers  # Normalize

#     # Find zero index dates in the outage data

#     df_outage_data = df_outage_data.groupby(df_outage_data.index).max()
#     zero_indices = df_outage_data.index[df_outage_data['sum'] == 0].tolist()
#     #print(zero_indices)
#     # Loop through each consecutive pair of zero indices
#     # Initialize list for no_outage events
#     no_outage_list = []

#     print("Finding timestamps with no outage events based on threshold.")

#     # Loop through each zero index and extract corresponding weather data
#     for zero_index in zero_indices:
#         zero_index_weather = pd.to_datetime(zero_index)

#         # Extract weather data for the zero outage period
#         max_ws = df_weather_data['sknt'].loc[zero_index_weather:zero_index_weather].max()
#         avg_ws = df_weather_data['sknt'].loc[zero_index_weather:zero_index_weather].mean()

#         max_g = df_weather_data['gust'].loc[zero_index_weather:zero_index_weather].max()
#         avg_g = df_weather_data['gust'].loc[zero_index_weather:zero_index_weather].mean()

#         pp_sum = df_weather_data['p01i'].loc[zero_index_weather:zero_index_weather].sum()

#         max_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].max()
#         min_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].min()
#         avg_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].mean()

#         year=df_weather_data['DATE'].loc[zero_index_weather].year
#         month=df_weather_data['DATE'].loc[zero_index_weather].month
#         day=df_weather_data['DATE'].loc[zero_index_weather].day


#         # Append to no_outage_list
#         no_outage_list.append({
#             'cust_out_max': 0,  # No customers affected
#             'out_duration_max': 0,
#             'area_cost_out_h': 0,
#             'area_KW_h': 0,
#             'max_wind_speed': max_ws,
#             'avg_wind_speed': avg_ws,
#             'impact_time': 0,
#             'outage_slope': 0,
#             'recovery_duration': 0,
#             'recovery_slope': 0,
#             'cust_normalized': 0,
#             'max_gust': max_g,
#             'avg_gust': avg_g,
#             'precipitation': pp_sum,
#             'Air_temp_max': max_tmpf,
#             'Air_temp_min': min_tmpf,
#             'Air_temp_avg': avg_tmpf,
#             'year':year,
#             'month':month,
#             'day':day,
#             'cummulative_customer_out' : 0


#         })

#     # Convert no_outage_list to DataFrame
#     no_outage_df = pd.DataFrame(no_outage_list)

#     #print(no_outage_df.head())
#     print("Finding outage events based on threshold.")

#     #no_outage_df = no_outage_df.drop(['run_start_time', 'sum','ws','N_sum'], axis =1)
#     event_data_list = []
#     for i in range(len(zero_indices) - 1):
#         first_zero_index = zero_indices[i]
#         second_zero_index = zero_indices[i + 1]

#         # Ensure the indices in both dataframes match before slicing
#         first_zero_index_weather = pd.to_datetime(first_zero_index)
#         second_zero_index_weather = pd.to_datetime(second_zero_index)

#         # Slicing the outage data for the current event
#         sliced_df = df_outage_data.loc[first_zero_index:second_zero_index].copy()
#         sliced_df.reset_index(drop=True, inplace=True)
#         sliced_df.index += 1  # Start index from 1
#         # sliced_df.index = range(1, len(sliced_df) + 1)
#         #print(sliced_df.head())
        
#         # Check if there are any non-zero values to process
#         # if (sliced_df['sum'].values > 0).any():
#         if (sliced_df['sum'] > 0).any():
#             sliced_df.loc[:,'run_start_time'] = pd.to_datetime(sliced_df['run_start_time'])
#             sliced_df.loc[:,'time_hours'] = (sliced_df['run_start_time'] - sliced_df['run_start_time'].iloc[0]).dt.total_seconds() / 3600
#             sliced_df.loc[:,'KW_out'] = (
#                 sliced_df['N_sum'] * 0.34 * 4.19 +
#                 sliced_df['N_sum'] * 0.35 * 23.91 +
#                 sliced_df['N_sum'] * 0.31 * 1301.0041
#             )
#             sliced_df.loc[:,'change'] = sliced_df['sum'].diff()
#             sliced_df.loc[:,'cummulative_customer_out'] = sliced_df.loc[sliced_df['change'] > 0, 'sum'].sum()



#             ''''
#             plt.figure(figsize=(12, 6))
#             plt.plot(sliced_df['time_hours'], sliced_df['sum'], label='Customer Outages', marker='o')
#             plt.plot(sliced_df['time_hours'], sliced_df['N_sum'], label='Normalized Customer Outages', marker='x')
#             plt.plot(sliced_df['time_hours'], sliced_df['KW_out'], label='Estimated KW Outage', linestyle='--')
    
#             plt.xlabel('Time since event start (hours)')
#             plt.ylabel('Outage Metrics')
#             plt.title('Outage Event Time Series')
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()
#             plt.show()
#            '''

#             # Check for the max index safely
#             if sliced_df['sum'].max() > 0:  # Ensure there is at least one non-zero value
#                 max_index = int(sliced_df['sum'].idxmax())
#                 impact_time = sliced_df['time_hours'].iloc[max_index] - sliced_df['time_hours'].iloc[0]
#                 outage_slope = (sliced_df['N_sum'].iloc[max_index] - sliced_df['N_sum'].iloc[0]) / impact_time
#                 recovery_duration = sliced_df['time_hours'].iloc[-1] - sliced_df['time_hours'].iloc[max_index]
#                 recovery_slope = sliced_df['N_sum'].iloc[max_index] / recovery_duration
#                 area_KW_h = np.trapezoid(sliced_df['KW_out'], sliced_df['time_hours'])
#                 cust_out_max = sliced_df['sum'].max()
#                 out_duration_max = sliced_df['time_hours'].max()
#                 area_cost_out = np.trapezoid(sliced_df['N_sum'], sliced_df['time_hours'])
#                 cust_normalized = sliced_df['N_sum'].max()
#                 cummulative_customer_out_max = sliced_df['cummulative_customer_out'].max()


#                 # Now use loc for slicing weather data for the current event
#                 max_ws = df_weather_data['sknt'].loc[first_zero_index_weather:second_zero_index_weather].max()
#                 avg_ws = df_weather_data['sknt'].loc[first_zero_index_weather:second_zero_index_weather].mean()

#                 max_g = df_weather_data['gust'].loc[first_zero_index_weather:second_zero_index_weather].max()
#                 avg_g = df_weather_data['gust'].loc[first_zero_index_weather:second_zero_index_weather].mean()

#                 pp_sum = df_weather_data['p01i'].loc[first_zero_index_weather:second_zero_index_weather].sum()

#                 max_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].max()
#                 min_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].min()
#                 avg_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].mean()

#                 year = df_weather_data['DATE'].loc[first_zero_index_weather].year
#                 month = df_weather_data['DATE'].loc[first_zero_index_weather].month
#                 day = df_weather_data['DATE'].loc[first_zero_index_weather].day

#                 event_data = {
#                     'cust_out_max': cust_out_max,
#                     'out_duration_max': out_duration_max,
#                     'area_cost_out_h': area_cost_out,
#                     'area_KW_h': area_KW_h,
#                     'max_wind_speed': max_ws,
#                     'avg_wind_speed': avg_ws,
#                     'impact_time': impact_time,
#                     'outage_slope': outage_slope,
#                     'recovery_duration': recovery_duration,
#                     'recovery_slope': recovery_slope,
#                     'cust_normalized': cust_normalized,
#                     'max_gust':max_g,
#                     'avg_gust':avg_g,
#                     'precipitation':pp_sum,
#                     'year': year,
#                     'month': month,
#                     'day': day,
#                     'Air_temp_max': max_tmpf,
#                     'Air_temp_min': min_tmpf,
#                     'Air_temp_avg': avg_tmpf,
#                     'cummulative_customer_out' : cummulative_customer_out_max


#                 }

#                 event_data_list.append(event_data)

#     # Convert event data list to DataFrame and save
#     print("Saving outage and non-outage events to parquet.")
#     event_df = pd.DataFrame(event_data_list)
#     event_df.to_parquet(f'Results/Outage_Events_Summary_All_{county}_{pct_threshold}_{start}-{end}.parquet', index=False)
#     #print(event_df.head())
#     #print(no_outage_df.head())
#     analysis_df = pd.concat([event_df, no_outage_df], ignore_index= True)
#     #print(analysis_df.head())
#     analysis_df.to_parquet(f'Results/Data_All_{county}_{pct_threshold}_{start}-{end}.parquet',index = False)


# Example
# state='Rhode Island'
# county='Bristol'
# start=2018
# end=2019
# main(state, county, start, end, 0.001)
