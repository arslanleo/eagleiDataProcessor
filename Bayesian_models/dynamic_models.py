
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

state='Washington'
county='Whatcom'
start=2018
end=2024

target_variable='tmpf'

# time series outage data
raw_outage_df=pd.read_excel(f'../outage_data/{state}/{county}/Merged_ZOH_Cleaned_data_{start}_{end}_{county}_{state}.xlsx')

# aggregated event data
event_df=pd.read_csv(f'../Results/Outage_Events_Summary_All_{county}_0.001_{start}-{end}.csv')

# weather data
weather_df=pd.read_csv(f'../weather_data/{state}/cleaned_weather_data_{county}.csv')

# 98th percentile of wind
percentile_98=weather_df[target_variable].quantile(.98)

# setting indices
raw_outage_df['run_start_time']=pd.to_datetime(raw_outage_df['run_start_time'])
raw_outage_df=raw_outage_df.set_index('run_start_time')
# filter weather data to be during the heat wave period
weather_df['valid']=pd.to_datetime(weather_df['valid'])
weather_df=weather_df.set_index('valid')

# investigate 98th percentile of wind events
event_df_filtered=event_df[event_df['Air_temp']>percentile_98].reset_index()

for i in range(len(event_df_filtered)):
    print(event_df_filtered.loc[i,'year'])
    start_date=datetime(event_df_filtered.loc[i,'year'],event_df_filtered.loc[i,'month'],event_df_filtered.loc[i,'day'])
    event_length=event_df_filtered.loc[i,'out_duration_max']
    end_date=start_date+pd.to_timedelta(event_length,unit='h')+pd.to_timedelta(24,unit='h')
    # filter outage and weather data to be during the event period
    selected_event_df=raw_outage_df.loc[start_date:end_date]
    selected_weather_df = weather_df.loc[start_date:end_date]

    # plotting
    fig, ax1 = plt.subplots()

    ax1.plot(selected_weather_df.index,selected_weather_df[target_variable],'r',label='Wind Gust (MPH)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Air Temperature (degrees F)',color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # ax3=ax1.twinx()
    # ax3.plot(selected_weather_df.index,selected_weather_df['p01i'],'g',label='Precipitation (inches)')
    # ax3.set_ylabel('Precipitation (in)',color='g')
    # ax3.tick_params(axis='y', labelcolor='g')

    ax2=ax1.twinx()
    ax2.step(selected_event_df.index,selected_event_df['sum'],'b', label='Num Customers Out')
    ax2.set_ylabel('Num Customers Out', color='b')
    fig.tight_layout()
    plt.title(f'Comparison of PNW Heat Wave to Event {i}, {county} County')
    plt.savefig(f'event_plots/{county}_event_{i}_{target_variable}.png')
    plt.close()
breakpoint()