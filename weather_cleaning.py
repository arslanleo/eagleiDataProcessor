import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

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

def find_county(file, state, county):
    # Event Datasets
    weather_dataset=pd.read_csv(file, low_memory=False)
    columns_to_keep = ['station', 'valid', 'tmpf', 'sknt', 'p01i', 'gust', 'lon', 'lat']
    weather_dataset = weather_dataset[columns_to_keep]

    # Load counties GeoJSON (replace 'ma_counties.geojson' with your file path)
    counties = gpd.read_file('gz_2010_us_050_00_5m.json', encoding='latin1')
    # ensure we are only looking in correct states (avoid duplicate county names and speed up process)
    county_to_fips=pd.read_csv('Eagle-idatasets/county_fips_master.csv', encoding='latin')
    ans=county_to_fips[county_to_fips['county_name']==f'{county} County']
    ans=ans[ans['state_name']==state]
    target_state_fips=round(ans['state'].values[0])
    counties['STATE']=pd.to_numeric(counties['STATE'], downcast='integer')
    counties.query('STATE==@target_state_fips', inplace=True)

    # Convert coordinates to gpd Points
    geometry = [Point(xy) for xy in zip(weather_dataset['lon'], weather_dataset['lat'])]
    stations_gdf = gpd.GeoDataFrame(weather_dataset, geometry=geometry)

    # Make sure both GeoDataFrames use the same coordinate reference system (CRS)
    # Usually GeoJSON is in EPSG:4326 (WGS84), so:
    counties = counties.to_crs(epsg=4326)
    stations_gdf = stations_gdf.set_crs(epsg=4326)

    # Spatial join to find which county each station falls into
    stations_with_county = gpd.sjoin(stations_gdf, counties, how='left', predicate='within')
    # Now stations_with_county has county info appended, e.g., 'NAME' or 'county' columns from GeoJSON

    print("Finding all weather stations in the associated county.")
    # return weather stations in county. If no weather stations in county, need
        # to return nearest county
    county_data=stations_with_county.query('NAME==@county')
    if county_data.empty:
        print(f"Warning: No weather stations in {county} county.")
        # find geometry of selected county and other weather stations in state
        list_of_points=stations_with_county['geometry'].unique()
        target_geometry=counties[counties['NAME']==county]['geometry'].values[0]

        # find nearest weather station in state
        distance=list()
        for point in list_of_points:
            distance.append(point.distance(target_geometry))
        min_index = distance.index(min(distance))
        closest_station=list_of_points[min_index]
        station_data=stations_with_county[stations_with_county['geometry']==closest_station]
        station=station_data['station'].unique()[0]
        closest_county=station_data['NAME'].unique()[0]
        print(f"Using nearest weather station to {county} county: {station} in {closest_county} County.")
        return station_data
    else:
        return county_data


# =====================================
# Preprocess each weather component
# =====================================
def main(state, county, start, end):

    # state='Florida'
    # county='Miami-Dade'
    file=f'weather_data/{state}/weather_{state}_{start}_{end}.csv'

    print('Sorting weather data at the county-level.')

    weather_data=find_county(file, state, county)

    print("Cleaning weather data.")
    #df_dew    = preprocess_weather_data(file,        'dwpf')
    df_gust   = preprocess_weather_data(weather_data,            'gust')
    df_sped   = preprocess_weather_data(weather_data,       'sknt')
    df_temp   = preprocess_weather_data(weather_data,            'tmpf')
    #df_pres   = preprocess_weather_data('weather_data_NOAA/asos_pressure.csv',        'mslp')
    #df_rh     = preprocess_weather_data('weather_data_NOAA/asos_RH.csv',              'relh')
    #df_wdir   = preprocess_weather_data('weather_data_NOAA/asos_wind_direction.csv',  'drct')
    df_precip = preprocess_weather_data(
        df=weather_data,
        column='p01i',
        replace_dict={'T': 0.001, 'M': 0},
        create_occurrence=True,
        drop_columns=['station', 'lon', 'lat'],
        keep_max=True
    )

    # =====================================
    # Merge all cleaned DataFrames on 'valid' timestamp
    # =====================================
    # Merge all DataFrames
    weather_dataset=pd.concat([df_temp['valid'],df_gust['gust'],df_sped['sknt'],df_precip['p01i'],df_temp['tmpf']],axis=1)
    # combined_df = df_temp.copy()
    # for df_component in [df_temp, df_gust, df_sped, df_precip]:
    #     combined_df = combined_df.merge(df_component, on='valid', how='inner')

    # Fill missing gust values with wind speed
    weather_dataset['gust'] = weather_dataset['gust'].fillna(weather_dataset['sknt'])
    print("Weather data has been successfully cleaned. Saving to csv.")
    # Preview
    #print(weather_dataset.head())
    weather_dataset.to_csv(f'weather_data/{state}/cleaned_weather_data_{county}.csv')


# use case example
# state='Rhode Island'
# county='Bristol'
# start=2018
# end=2019
# main(state, county, start, end)