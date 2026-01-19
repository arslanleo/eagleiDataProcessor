"""
Example script that scrapes data from the IEM ASOS download service.

More help on CGI parameters is available at:

    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help

Requires: Python 3
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from urllib.request import urlopen
import pandas as pd
from io import StringIO
import geopandas as gpd

# Number of attempts to download data
MAX_ATTEMPTS = 6
# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"


def download_data(uri):
    """Fetch the data from the IEM

    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.

    Args:
      uri (string): URL to fetch

    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print(f"download_data({uri}) failed with {exp}")
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def get_stations_from_filelist(filename):
    """Build a listing of stations from a simple file listing the stations.

    The file should simply have one station per line.
    """
    if not os.path.isfile(filename):
        print(f"Filename {filename} does not exist, aborting!")
        sys.exit()
    with open(filename, encoding="ascii") as fh:
        stations = [line.strip() for line in fh]
    return stations

def find_stations_in_county(state, county, jdict):
    # Load counties GeoJSON (replace 'ma_counties.geojson' with your file path)
    counties = gpd.read_file('gz_2010_us_050_00_5m.json', encoding='latin1')
    # ensure we are only looking in correct states (avoid duplicate county names and speed up process)
    county_to_fips=pd.read_csv('Eagle-idatasets/county_fips_master.csv', encoding='latin')
    ans=county_to_fips[county_to_fips['county_name']==f'{county} County']
    ans=ans[ans['state_name']==state]
    target_state_fips=round(ans['state'].values[0])
    # ---- Filter counties early ----
    counties = counties.copy()
    counties['STATE'] = counties['STATE'].astype(int)
    counties = counties.query('STATE == @target_state_fips')
    target_county = counties.query('NAME == @county')

    if target_county.empty:
        raise ValueError("County not found")

    stations=[]
    lon=[]
    lat=[]
    for site in jdict["features"]:
        stations.append(site["properties"]["sid"]) # noqa
        lon.append(site["geometry"]["coordinates"][0])
        lat.append(site["geometry"]["coordinates"][1])

    # Create dataframe
    df=pd.DataFrame({
        "station":stations,
        "lon":lon,
        "lat":lat
    })
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"  # WGS84 lon/lat
    )

    # ---- Spatial join only with target county ----
    joined = gpd.sjoin(
        gdf,
        target_county,
        how="inner",
        predicate="within"
    )

    if not joined.empty:
        return joined

    # ---- No stations in county → find nearest ----
    print(f"Warning: No weather stations in {county} county.")

    # Project for distance calculations
    stations_proj = gdf.to_crs(epsg=5070)
    county_proj = target_county.to_crs(epsg=5070)

    # Use spatial index nearest
    nearest_idx = stations_proj.sindex.nearest(
        county_proj.geometry.iloc[0],
        return_all=False
    )[1][0]

    station_data = gdf.iloc[[nearest_idx]]

    print(
        f"Using nearest weather station to {county} county: "
        f"{station_data['station'].iloc[0]}"
    )

    return station_data

def get_stations_from_networks(state, county):
    """Build a station list by using a bunch of IEM networks."""
    # stations = []
    state_to_code=pd.read_csv('eagle-idatasets/county_fips_master.csv', encoding='latin')
    result = state_to_code[state_to_code['state_name'] == state]
    state_abbr = result['state_abbr'].values[0]
    network=f"{state_abbr}_ASOS"

    # Get metadata
    uri = (
        "https://mesonet.agron.iastate.edu/"
        f"geojson/network/{network}.geojson"
    )
    data = urlopen(uri)
    jdict = json.load(data)
    # for site in jdict["features"]:
    #     stations.append(site["properties"]["sid"])  # noqa
    joined=find_stations_in_county(state, county, jdict)
    stations=list(joined['station'])
    return stations

#
# def download_alldata():
#     """An alternative method that fetches all available data.
#
#     Service supports up to 24 hours worth of data at a time."""
#     # timestamps in UTC to request data for
#     startts = datetime(2012, 8, 1)
#     endts = datetime(2012, 9, 1)
#     interval = timedelta(hours=24)
#
#     service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"
#
#     now = startts
#     while now < endts:
#         thisurl = service
#         thisurl += now.strftime("year1=%Y&month1=%m&day1=%d&")
#         thisurl += (now + interval).strftime("year2=%Y&month2=%m&day2=%d&")
#         print(f"Downloading: {now}")
#         data = download_data(thisurl)
#         outfn = f"{now:%Y%m%d}.txt"
#         with open(outfn, "w", encoding="ascii") as fh:
#             fh.write(data)
#         now += interval


def main(state, county, start, end, outfile):
    """Our main method"""
    print("Began process of fetching weather data from IEM servers.")
    # timestamps in UTC to request data for
    startts = datetime(start, 1, 1)
    endts = datetime(end+1, 1, 1)   # the last day is exclusive so adding 1 to end year

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
    service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

    stations = get_stations_from_networks(state, county)
    print(f"Found {len(stations)} stations for {county} County, {state}.")
    data=''
    for i, station in enumerate(stations):
        uri = f"{service}&station={station}"
        print(f"Downloading weather data from station {station} ({i+1}/{len(stations)})")
        new_data = download_data(uri)
        # for debugging purposes, to make sure that there is enough data downloaded for each station
        print(f"Downloaded {len(new_data)} bytes of data from station {station}")
        data += new_data
    print(f"Weather data has been downloaded for {state}. Saving to parquet: {outfile}")
    
    data=StringIO(data)
    # change to csv format
    data=pd.read_csv(data, sep=',', comment='#', low_memory=False)
    # filter out repeated columns
    data = data[data['tmpf'] != 'tmpf']

    data.to_parquet(outfile)

# main('Rhode Island',2018,2019)
# #
# if __name__ == "__main__":
#     download_alldata()