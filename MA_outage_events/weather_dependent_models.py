from tqdm import tqdm
import pandas as pd
import numpy as np 
import ast

event_dataset = pd.read_csv('event_dataset_with_stations.csv')
weather_dataset = pd.read_csv('clean_weather_data_all.csv')

def safe_literal_eval(val):
    if pd.isna(val):
        return []  # or np.nan or any default you want
    else:
        return ast.literal_eval(val)

event_dataset['station'] = event_dataset['station'].apply(safe_literal_eval)

weather_dataset['valid'] = pd.to_datetime(weather_dataset['valid'])
event_dataset['event_start_time'] = pd.to_datetime(event_dataset['event_start_time'])
event_dataset['event_end_time'] = pd.to_datetime(event_dataset['event_end_time'])

weather_cols_to_max = ['tmpf', 'sknt', 'gust', 'gust_plus_speed', 'p01i']

max_weather_data = {col: [] for col in weather_cols_to_max}

for _, event_row in tqdm(event_dataset.iterrows(), total=len(event_dataset), desc="Processing events"):
    stations = event_row['station']  # list of stations for this event's county
    start_time = event_row['event_start_time']
    end_time = event_row['event_end_time']

    mask = (
        (weather_dataset['station'].isin(stations)) &
        (weather_dataset['valid'] >= start_time) &
        (weather_dataset['valid'] <= end_time)
    )
    filtered_weather = weather_dataset[mask]

    for col in weather_cols_to_max:
        if not filtered_weather.empty:
            max_val = filtered_weather[col].max()
        else:
            max_val = pd.NA
        max_weather_data[col].append(max_val)

for col in weather_cols_to_max:
    event_dataset[f'max_{col}'] = max_weather_data[col]

print(event_dataset.head())
event_dataset.to_csv('final_data.csv', index=False)

print(event_dataset[['event_id', 'county', 'max_tmpf', 'max_sknt', 'max_gust', 'max_gust_plus_speed']].head())
