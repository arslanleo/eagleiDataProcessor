# EAGLE-I Data Processor

## Overview
This project processes and aligns power outage data from the EAGLE-I dataset with ASOS weather data to enable advanced analysis of weather-driven outage events. The pipeline is designed for researchers and data analysts studying the relationship between weather conditions and power system resilience and reliability.

This repository contains scripts for:
- Cleaning and Preprocessing EAGLE-I Data
- Identifying Events in the EAGLE-I Data
- Downloading and Cleaning Weather Data
- Merging the EAGLE-I outage data and Weather data to form a single dataset

## How to use
### Initial Setup (First Time)

#### 1. Installation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd eagleiDataProcessor
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

#### 2. Prepare Data
- Ensure EAGLE-I CSV files are in `Eagle-idatasets/` folder
- Files should follow pattern: `eaglei_outages_{YEAR}.csv`

#### 3. Update Configuration
Edit `config.json` with your analysis parameters:
```json
{
  "analysis_parameters": {
    "state": "YourState",
    "county": "YourCounty",
    "start_year": 2020,
    "end_year": 2024
  }
}
```

---

### Running the Pipeline

#### Option 1: Full Pipeline (Recommended for First Run)
```bash
python main.py --mode full --config config.json
```
This runs all 4 stages: fetch weather → clean weather → clean outage → align data

#### Option 2: Individual Stages
```bash
# Just fetch weather
python main.py --mode weather-only

# Just align data (if already have cleaned data)
python main.py --mode alignment-only

# Just process outages
python main.py --mode outage-only
```

#### Option 3: Override Configuration
```bash
python main.py \
  --state "Florida" \
  --county "Miami-Dade" \
  --start-year 2020 \
  --end-year 2024
```

---

### Monitoring Progress

1. **View logs:**
   ```bash
   tail -f logs/eaglei_dataprocessor.log
   ```

2. **Check configuration before running:**
   ```bash
   python main.py --show-config
   ```

3. **Enable debug logging:**
   ```bash
   python main.py --log-level DEBUG
   ```

---

## Workflow

The pipeline follows a 4-stage process:

### Stage 1: Fetch Weather Data
- Automatically retrieves ASOS weather data from IEM servers
- Stores data in parquet format for efficient processing
- Skipped if data already exists

### Stage 2: Clean Weather Data
- Interpolates missing values using forward fill
- Standardizes timestamps to 15-minute intervals
- Extracts key weather variables, as specified in the `config` file
- Save the data from all the weather stations in a county as a hierarchical NetCDF file

### Stage 3: Clean Outage Data
- Filters data for specified state and county
- Fills missing customer outage values
- Merges data across specified year range
- Extract events from the data
- Save the cleaned data in parquet formal

### Stage 4: Align and Merge
- Temporally aligns and merge the outage and weather datasets
- Stores the merged data in NetCDF format in the `merged_data` folder 

## Configuration Parameters:

**Basic Parameters (Processing Parameters):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | string | State name for analysis |
| `county` | string | County name for analysis |
| `start_year` | int | Start year (inclusive) |
| `end_year` | int | End year (inclusive) |

**Processing Options:**
| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `fetch_weather_data` | bool | Automatically fetch weather data from IEM | true |
| `clean_weather_data` | bool | Clean and preprocess weather data | true |
| `clean_outage_data` | bool | Clean and preprocess outage data | true |
| `align_data` | bool | Align and merge outage-weather datasets | true |
| `skip_if_exists` | bool | Skip processing if an output already exists | true |

**Weather Variables:**
The `weather_variables` key in the config file can be expanded to include other weather variables

**Advanced Parameters (Data Cleaning Parameters):**
| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| `min_customers_before_gap` | int | Criteria to select EAGLE-I data gaps for filling, only select data gaps that have this many minimum customers before the gap | 20 |
| `min_customers_after_gap` | int | Criteria to select EAGLE-I data gaps for filling, only select data gaps that have this many minimum customers after the gap | 2 |
| `max_gap_minutes` | int | Criteria to select EAGLE-I data gaps for filling, only select data gaps that have atleast this much duration | 1440 |
| `use_auto_gap_rank_threshold` | bool |Automatically decide a threshold for the rank metric to fill all the data gaps above that threshold | true |
| `gap_rank_threshold_quantile` | float | Instead of automatic selection, select a rank threshold based on this quantile (set `use_auto_gap_rank_threshold` = false for this to work) | 0.4 |
| `events_customer_threshold` | int | Threshold for event identification (further explanation below).  | 30 |


## How the Events are defined and extracted from the EAGLE-I Data
- The Events in the EAGLE-I data are defined for each county separately. These are called the "county events".
- Once data cleaning is complete, the EAGLE-I data is sorted in ascending order by timestamp.
- The timeseries of `customers_out` and the timestamps in the EAGLE-I data are scanned, and whenever the `customers_out` value goes from less than the `events_customer_threshold` (defined in the config file) to **greater than or equal** to it, a new event is started.
- When the `customers_out` value goes back to **less than** the `events_customer_threshold`, the event is considered ended, and a unique event number is assigned to it.
- Once EAGLE-I event processing is completed, the EAGLE-I dataframe is returned with an events_number column appended.
- **Note**: The event column contains unique event numbers assigned to the events defined as explained above, but the rest of the data (between two consecutive events) also remains in the dataset, with different event numbers assigned to it. That data is kept for further analysis of timestamps that don't fall under the events defined above. Those values can be easily excluded by grouping the dataframe by the events column and selecting events with a minimum value of `customers_out` equal to or greater than the `events_customer_threshold`.


## Folder Structure
- **eagle-idatasets/**: Raw outage data for all counties.
- **outage_data/**: Folder where cleaned outage data is stored after preprocessing
- **weather_data/**: Contains both cleaned and uncleaned ASOS weather data
- **merged_data/**: Contains the final merged otuput of weather and EAGLE-I outage data
- **results/**: Output results of any analysis can be output to this folder
- **docs/**: Contains helpful documents
- **logs/**: Contains logs
- **misc/**: Contains miscellaneous input/output file
- **src/**: Contains the source code

## Scripts
- **`config.json`**  
  This is the configuration file that the user can modify to provide inputs about which counties and states to process.  
  It additionally contains all the other parameters and configurations which can be modified as required. 

- **`main.py`**  
  The driver of the other functions - put your inputs in here and it will automatically do the full outage-weather data extraction + aggregation process.

- **`outage_cleaning.py`**  
  Cleans the Eagle-I outage data and stores the cleaned results in the `outage_data/` folder.

- **`weather_cleaning.py`**  
  Cleans the ASOS weather data and stores the output in the `weather_data/` folder.

- **`merge_outage_weather.py`**  
  Merge the cleaned outage and weather datasets.

- **`fetch_weather_data.py`**
  Automatically scrapes ASOS website for state weather data within given years and outputs it into weather_data folder.

- **`analysis_examples.ipynb`**
  Jupyter Notebook that contain examples on how to access different variables in the final merged dataset and how to analyze events.

## Data Sources

- **Outage Data:** EAGLE-I Initiative (https://www.eagle-i.gov/)
- **Weather Data:** ASOS Network via Iowa Environmental Mesonet (https://mesonet.agron.iastate.edu/)

## Acknowledgments

- EAGLE-I Initiative for outage data
- Iowa Environmental Mesonet for weather data

---

**Last Updated:** January 2026
**Version:** 1.0.0