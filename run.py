#---------------------------------------------------------------------------------------#
#  run.py:
#  this function goes through the process of cleaning all outage and weather data from a specific time period
#  and merging it into one large dataset.

# packages
import pandas as pd

# inputs

state="Texas"
county='Harris'
start='2018'
end='2024'
# Load MCC Data
pdf = pd.read_csv('MCC.csv')
# target_fips = '17031'  # For Cook County
#target_fips = '4013' # maricopa
#target_fips = '25025' # suffolk
#target_fips = '12086' # miami-dade
#target_fips = '53033' # king
target_fips = '48201' # harris
result = pdf[pdf['County_FIPS'] == target_fips]
customers = result['Customers'].values