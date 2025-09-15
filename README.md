# Outage_Analysis_eaglei

This repository contains outage and weather data processing scripts using the Eagle-I dataset, focused on outage analysis across all counties.

## Folder Structure
- **Eaglei_data/**: Raw outage data for all counties.
- **outage_data/**: Folder where cleaned outage data is stored after preprocessing.
- **weather_data/**: Contains both cleaned and uncleaned ASOS weather data
- **Results/**: Output results from run.py are stored here.

## Scripts

- **`run.py`**
  The driver of the other functions - put your inputs in here and it will automatically do the full outage-weather data extraction + aggregation process.

- **`outage_cleaning.py`**  
  Cleans the Eagle-I outage data and stores the cleaned results in the `outage_data/` folder.

- **`weather_cleaning.py`**  
  Cleans the ASOS weather data and stores the output in the `weather_data/` folder.

- **`map_outage_weather.py`**  
  Maps the cleaned outage and weather datasets to extract event-based summaries used for downstream analysis.

- **`fetch_weather_data.py`**
  Automatically scrapes ASOS website for state weather data within given years and outputs it into weather_data folder.

## Purpose

This code is used to preprocess and align outage and weather data, generate clean event features, and prepare the dataset for modeling or analysis related to outage prediction and impact studies.
