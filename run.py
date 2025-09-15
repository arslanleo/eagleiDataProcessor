#---------------------------------------------------------------------------------------#
#  run.py:
#  this function goes through the process of cleaning all outage and weather data from a specific time period
#  and merging it into one large dataset.
#
# BEFORE RUNNING: Insert EAGLE-I datasets into Eagle-idatasets folder

# packages
import pandas as pd
import weather_cleaning
import map_outage_weather
import fetch_weather_data
import outage_cleaning

# inputs

state="Texas"
county='Harris'
start='2018'
end='2024'

# fetch weather data
print("Starting Outage + Weather Process.")

print("Fetching and saving weather data.")
fetch_weather_data.get_stations_from_networks()
# clean + map weather data
print("Done fetching and saving weather data.")
print("Cleaning and filtering weather data.")
weather_cleaning.run_weather_cleaning(state, county)
# clean outage data
print("Done cleaning and filtering weather data.")
print("Cleaning outage data.")
outage_cleaning.outage_cleaning(state, county, start, end)
# merge outage and weather data
print("Done cleaning outage data.")

print("Aggregating outage and weather data.")
map_outage_weather.outage_weather_agg(state, county, start, end)
print("Done! Please see all data "
      "and event only data in results folder.")