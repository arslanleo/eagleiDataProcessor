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
state="Texas"
county='Harris'
start=2018
end=2024
threshold=0.001


# ensure proper file paths exist
output_folder = 'cleaned_data/'
os.makedirs(output_folder, exist_ok=True)
output_folder = 'Results/'
os.makedirs(output_folder, exist_ok=True)
output_folder = 'weather_data/'
os.makedirs(output_folder, exist_ok=True)
# fetch weather data
fetch_weather_data.main(state, start, end)
# clean + map weather data
weather_cleaning.main(state, county)
# clean outage data
outage_cleaning.main(state, county, start, end)
# merge outage and weather data
map_outage_weather.main(state, county, start, end, threshold)
print("Done! Please see all data "
      "and event only data in results folder.")