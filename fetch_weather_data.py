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


def get_stations_from_networks(state):
    """Build a station list by using a bunch of IEM networks."""
    stations = []
    state_to_code=pd.read_csv('Eagle-idatasets/county_fips_master.csv', encoding='latin')
    result = state_to_code[state_to_code['state_name'] == state]
    state_abbr = result['state_abbr'].values[0]
    # states = (
    #     "AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN "
    #     "MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT "
    #     "WA WI WV WY"
    # )
    network=f"{state_abbr}_ASOS"
    #networks = [f"{state}_ASOS" for state in states.split()]

    # Get metadata
    uri = (
        "https://mesonet.agron.iastate.edu/"
        f"geojson/network/{network}.geojson"
    )
    data = urlopen(uri)
    jdict = json.load(data)
    for site in jdict["features"]:
        stations.append(site["properties"]["sid"])  # noqa
    return stations


def download_alldata():
    """An alternative method that fetches all available data.

    Service supports up to 24 hours worth of data at a time."""
    # timestamps in UTC to request data for
    startts = datetime(2012, 8, 1)
    endts = datetime(2012, 9, 1)
    interval = timedelta(hours=24)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    now = startts
    while now < endts:
        thisurl = service
        thisurl += now.strftime("year1=%Y&month1=%m&day1=%d&")
        thisurl += (now + interval).strftime("year2=%Y&month2=%m&day2=%d&")
        print(f"Downloading: {now}")
        data = download_data(thisurl)
        outfn = f"{now:%Y%m%d}.txt"
        with open(outfn, "w", encoding="ascii") as fh:
            fh.write(data)
        now += interval


def main(state, start, end):
    """Our main method"""
    print("Began process of fetching weather data from IEM servers.")
    # timestamps in UTC to request data for
    startts = datetime(start, 1, 1)
    endts = datetime(end, 12, 31)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
    service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

    stations = get_stations_from_networks(state)
    print(f"Found {len(stations)} stations for {state}.")
    data=''
    for i, station in enumerate(stations):
        uri = f"{service}&station={station}"
        print(f"Downloading weather data from station {station} ({i+1}/{len(stations)})")
        data = data + download_data(uri)
    print(f"Weather data has been downloaded for {state}. Saving to csv.")
    
    data=StringIO(data)
    # change to csv format
    data=pd.read_csv(data, sep=',', comment='#', low_memory=False)
    # filter out repeated columns
    data = data[data['tmpf'] != 'tmpf']
    outfile=f'weather_data/{state}/weather_{state}_{start}_{end}.csv'
    data.to_csv(outfile)
    # outfn = f"{station}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}.txt"
    # with open(outfn, "w", encoding="ascii") as fh:
    #     fh.write(data)

# main('Rhode Island',2018,2019)
# #
# if __name__ == "__main__":
#     download_alldata()