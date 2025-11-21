#---------------------------------------------------------------------------------------#
#  run.py:
#  this function goes through the process of cleaning all outage and weather data from a specific time period
#  and merging it into one large dataset.
#
# BEFORE RUNNING: Insert EAGLE-I datasets into Eagle-idatasets folder
#----------------------------------------------------------------------------------------------

# packages
import pandas as pd
import weather_cleaning
import map_outage_weather
import fetch_weather_data
import outage_cleaning
import os

# inputs
state="Washington"
county='Pierce'
start=2018
end=2019    #2024
threshold=0.001


# ensure proper file paths exist
output_folder = f'outage_data/{state}/{county}'
os.makedirs(output_folder, exist_ok=True)
output_folder = 'Results/'
os.makedirs(output_folder, exist_ok=True)
output_folder = f'weather_data/{state}'
os.makedirs(output_folder, exist_ok=True)

# fetch weather data

# add if statement that only fetches weather data if its not already loaded in
a=f"weather_data/{state}/weather_{state}_{start}_{end}.csv"
if os.path.isfile(a):
    print("State weather data already downloaded. Moving onto weather cleaning.")
else:
    print("State weather data not found. Proceeding to fetch weather from IEM servers.")
    fetch_weather_data.main(state, start, end)

# clean + map weather data
a=f"weather_data/{state}/cleaned_weather_data_{county}.csv"
if os.path.isfile(a):
    print("Weather Data already cleaned. Moving onto outage cleaning.")
else:
    weather_cleaning.main(state, county, start, end)

# clean outage data
a = f"outage_data/{state}/{county}/Merged_ZOH_Cleaned_data_{start}_{end}_{county}_{state}.xlsx"
if os.path.isfile(a):
    print("Outage Data already cleaned. Moving onto merge process.")
else:
    outage_cleaning.main(state, county, start, end)

# merge outage and weather data
map_outage_weather.main(state, county, start, end, threshold)
print("Done! Please see the combined outage-weather data "
      "in results folder.")