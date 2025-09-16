import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# =====================================
# Define the common 15-minute timestamp range
# =====================================
start = pd.Timestamp('2018-01-01 00:15:00')
end = pd.Timestamp('2024-12-30 23:45:00')
full_time_range = pd.date_range(start=start, end=end, freq="15min")
full_df = pd.DataFrame({"valid": full_time_range})

# =====================================
# Function to preprocess individual weather datasets
# =====================================
def preprocess_weather_data(df, column, replace_dict=None, create_occurrence=False, keep_max=True, drop_columns=None):
    df["valid"] = pd.to_datetime(df["valid"]).dt.round("15min")

    if replace_dict:
        df[column] = df[column].replace(replace_dict)

    df[column] = pd.to_numeric(df[column], errors='coerce')

    if keep_max:
        df = df.sort_values(by=["valid", column], ascending=[True, False])
        df = df.groupby("valid").first().reset_index()

    df = full_df.merge(df, on="valid", how="left")

    df[column] = df[column].interpolate(method="nearest")
    df.ffill(inplace=True)

    # ALWAYS drop problematic columns if they exist
    for col in ['station', 'lat', 'lon']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    if drop_columns:
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    if create_occurrence:
        df["poccurence"] = df[column].apply(lambda x: 0 if x == 0 else 1)

    return df

def find_county(file, county):
    # Event Datasets
    weather_dataset=pd.read_csv(file)
    columns_to_keep = ['station', 'valid', 'tmpf', 'sknt', 'p01m', 'gust', 'lon', 'lat']

    weather_dataset = weather_dataset[columns_to_keep]

    # Load counties GeoJSON (replace 'ma_counties.geojson' with your file path)
    counties = gpd.read_file('gz_2010_us_050_00_5m.json', encoding='latin1')

    geometry = [Point(xy) for xy in zip(weather_dataset['lon'], weather_dataset['lat'])]
    stations_gdf = gpd.GeoDataFrame(weather_dataset, geometry=geometry)

    # Make sure both GeoDataFrames use the same coordinate reference system (CRS)
    # Usually GeoJSON is in EPSG:4326 (WGS84), so:
    counties = counties.to_crs(epsg=4326)
    stations_gdf = stations_gdf.set_crs(epsg=4326)

    # Spatial join to find which county each station falls into
    stations_with_county = gpd.sjoin(stations_gdf, counties, how='left', predicate='within')

    # Now stations_with_county has county info appended, e.g., 'NAME' or 'county' columns from GeoJSON

    # return target county
    return stations_with_county.query('NAME==@county')


# =====================================
# Preprocess each weather component
# =====================================
def main(state, county, start, end):

    # state='Florida'
    # county='Miami-Dade'
    file=f'weather_data/{state}/weather_{state}_{start}_{end}.csv'

    print('Filtering by selected county.')

    weather_data=find_county(file, county)

    print("Processing weather data.")
    #df_dew    = preprocess_weather_data(file,        'dwpf')
    df_gust   = preprocess_weather_data(weather_data,            'gust')
    df_sped   = preprocess_weather_data(weather_data,       'sknt')
    df_temp   = preprocess_weather_data(weather_data,            'tmpf')
    #df_pres   = preprocess_weather_data('weather_data_NOAA/asos_pressure.csv',        'mslp')
    #df_rh     = preprocess_weather_data('weather_data_NOAA/asos_RH.csv',              'relh')
    #df_wdir   = preprocess_weather_data('weather_data_NOAA/asos_wind_direction.csv',  'drct')
    df_precip = preprocess_weather_data(
        df=weather_data,
        column='p01m',
        replace_dict={'T': 0.001, 'M': 0},
        create_occurrence=True,
        drop_columns=['station', 'lon', 'lat'],
        keep_max=True
    )

    # =====================================
    # Merge all cleaned DataFrames on 'valid' timestamp
    # =====================================
    # Merge all DataFrames
    print("Merging weather datasets.")
    weather_dataset=pd.concat([df_temp['valid'],df_gust['gust'],df_sped['sknt'],df_precip['p01m'],df_temp['tmpf']],axis=1)
    # combined_df = df_temp.copy()
    # for df_component in [df_temp, df_gust, df_sped, df_precip]:
    #     combined_df = combined_df.merge(df_component, on='valid', how='inner')

    # Fill missing gust values with wind speed
    weather_dataset['gust'] = weather_dataset['gust'].fillna(weather_dataset['sknt'])

    # Preview
    #print(weather_dataset.head())
    weather_dataset.to_csv(f'weather_data/{state}/cleaned_weather_data_{county}.csv')
