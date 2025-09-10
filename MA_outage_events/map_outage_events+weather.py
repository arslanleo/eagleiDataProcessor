import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

#Event Datasets 
event_dataset = pd.read_csv('county_event_dataset.csv')

event_dataset['event_start_time'] = pd.to_datetime(event_dataset['event_start_time'])
event_dataset['event_end_time'] = pd.to_datetime(event_dataset['event_end_time'])

# Calculate duration as timedelta
event_dataset['event_duration'] = event_dataset['event_end_time'] - event_dataset['event_start_time']

# Optionally, convert duration to total hours (float)
event_dataset['event_duration_hours'] = event_dataset['event_duration'].dt.total_seconds() / 3600

print(event_dataset.head())

weather_dataset = pd.read_csv('asos_clean.csv')
print(weather_dataset.head())
print(weather_dataset.columns)
columns_to_keep = ['station', 'valid', 'tmpf', 'sknt', 'p01i', 'gust','lon','lat']

weather_dataset = weather_dataset[columns_to_keep]
print(weather_dataset.head())

weather_dataset= weather_dataset.replace("M", 0)
cols_to_convert = ['tmpf', 'sknt', 'p01i', 'gust']
for col in cols_to_convert:
    weather_dataset[col] = pd.to_numeric(weather_dataset[col], errors='coerce')

# Set p01i to 1 if not zero, else 0
weather_dataset['p01i'] = weather_dataset['p01i'].apply(lambda x: 1 if x != 0 else 0)

# Create new column gust_plus_speed = max of gust and sknt
weather_dataset['gust_plus_speed'] = weather_dataset[['gust', 'sknt']].max(axis=1)

# Check result
print(weather_dataset.head())
weather_dataset.to_csv('clean_weather_data_all.csv', index=False)
# Load counties GeoJSON (replace 'ma_counties.geojson' with your file path)
counties = gpd.read_file('../gz_2010_us_050_00_5m.json', encoding='latin1')
print(counties.head())

geometry = [Point(xy) for xy in zip(weather_dataset['lon'], weather_dataset['lat'])]
stations_gdf = gpd.GeoDataFrame(weather_dataset, geometry=geometry)

# Make sure both GeoDataFrames use the same coordinate reference system (CRS)
# Usually GeoJSON is in EPSG:4326 (WGS84), so:
counties = counties.to_crs(epsg=4326)
stations_gdf = stations_gdf.set_crs(epsg=4326)

# Spatial join to find which county each station falls into
stations_with_county = gpd.sjoin(stations_gdf, counties, how='left', predicate='within')

# Now stations_with_county has county info appended, e.g., 'NAME' or 'county' columns from GeoJSON
print(stations_with_county[['station', 'lon', 'lat', 'NAME']].head())

stations_with_county.drop(columns='geometry').to_csv('stations_with_county.csv', index=False)
event_counties = set(event_dataset['county'].unique())

# Filter stations to only those counties
stations_in_event_counties = stations_with_county[stations_with_county['NAME'].isin(event_counties)]

# Group stations by county and list unique station names
stations_by_county = stations_in_event_counties.groupby('NAME')['station'].unique().reset_index()


stations_by_county['station'] = stations_by_county['station'].apply(list)

print(stations_by_county)

station_data = {
    'county': ['Barnstable', 'Berkshire', 'Bristol', 'Dukes', 'Essex', 'Franklin', 'Hampden', 'Middlesex', 'Nantucket', 'Norfolk', 'Plymouth', 'Suffolk', 'Worcester'],
    'station': [
        ['PVC', 'FMH', 'CQX', 'HYA'],
        ['AQW', 'PSF'],
        ['TAN', 'EWB'],
        ['MVY'],
        ['BVY', 'LWM'],
        ['ORE'],
        ['BAF', 'CEF'],
        ['BED'],
        ['ACK'],
        ['OWD', 'MQE'],
        ['GHG', 'PYM'],
        ['BOS'],
        ['FIT', 'ORH']
    ]
}

stations_df = pd.DataFrame(station_data)


event_dataset = event_dataset.merge(stations_df, how='left', left_on='county', right_on='county')

# Check result
print(event_dataset[['county', 'station']].head())
event_dataset.to_csv('event_dataset_with_stations.csv', index=False)