import matplotlib.pyplot as plt
import pandas as pd

# inputs - change as needed
state='Washington'
county='King'
start='2018'
end='2024'
threshold=0.001

def plot_wind(df):
    y, x = plt.hist(df["gust"].dropna(), bins=30, color="royalblue", edgecolor="black", alpha=0.7,density=True)
    plt.xlabel("Wind Gust + Speed (mph)")
    plt.ylabel("Probability")
    plt.title("Histogram of Wind Gust")
    plt.show()
def plot_precip(df):
    df["poccurance"] = df["precipitation"].apply(lambda x: 0 if x == 0 else 1)
    plt.hist(df["poccurance"].dropna(), bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Precipitation")
    plt.ylabel("Frequency")
    plt.title("Histogram of Precipitation")
    plt.show()

def plot_max_temp(df):
    plt.hist(df["Air_temp"].dropna(), bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Maximum Temp")
    plt.ylabel("Frequency")
    plt.title("Histogram of Maximum Temperature")
    plt.show()

def plot_min_temp(df):
    plt.hist(df["Air_temp_min"].dropna(), bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Minimum Temp")
    plt.ylabel("Frequency")
    plt.title("Histogram of Minimum Temperature")
    plt.show()

def plot_seasonal_outages(df):
    plt.hist(df["month"])
    plt.xlabel("Month")
    plt.ylabel("Frequency")
    plt.title("Seasonal Outage Frequency")
    plt.show()

def plot_seasonal_weather_behavior(df):
    plt.scatter(df['month'], df['gust'])
    plt.scatter(df['month'], df['precipitation'])
    plt.scatter(df['month'],df['Air_temp'])
    plt.legend(['Gust', 'Precipitation', 'Temperature'])
    plt.show()

def outage_mag_by_month(df):
    outage_event_df = df.groupby('month').agg({'cust_normalized': 'mean'})
    plt.bar(outage_event_df.index, outage_event_df['cust_normalized'])
    plt.xlabel('month')
    plt.ylabel('Normalized Customer Out')
    plt.show()

def outage_duration_by_month(df):
    outage_event_df = df.groupby('month').agg({'out_duration_max': 'mean'})
    plt.bar(outage_event_df.index, outage_event_df['out_duration_max'])
    plt.xlabel('month')
    plt.ylabel('Outage Duration')
    plt.show()

def myround(x, base=0.05):
    return base * round(x/base)

def compare_counties(df1,df2,target_variable):

    df2[target_variable]=myround(df1[target_variable])
    df2[target_variable] = myround(df2[target_variable])

    # average all outage instances over their target weather variable
    df1 = df1.groupby(target_variable).agg({

        'cummulative_customer_out': 'median',
        'cust_normalized': 'median'

    }).reset_index()

    # average all outage instances over their target weather variable
    df2 = df2.groupby(target_variable).agg({

        'cummulative_customer_out': 'median',
        'cust_normalized': 'median'

    }).reset_index()
    plt.scatter(df1[target_variable],df1['cust_normalized'], label='King County')
    plt.scatter(df2[target_variable],df2['cust_normalized'],label='Maricopa County')

    plt.xlabel(f'{target_variable}')
    plt.ylabel('Portion of Customers Out')
    plt.title('County Comparison in 97.5th percentile of King County Weather')
   # plt.yscale('log')
    plt.legend()
    plt.show()

# read in data
outage_event_df=pd.read_csv(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.csv')
all_data=pd.read_csv(f'../Results/Data_All_{county}_{threshold}_{start}-{end}.csv')
all_data1=pd.read_csv(f'../Results/Data_All_Maricopa_{threshold}_{start}-{end}.csv')
target_variable='precipitation'
# identify 97.5th percentile of 5 or more days for heat wave
percentile_975=all_data1[target_variable].quantile(.975)
filtered_data=outage_event_df[outage_event_df[target_variable]>percentile_975]
filtered_data=filtered_data.sort_values(by=['year', 'month', 'day'], ascending=[True, True, True])

outage_event_df_1=pd.read_csv(f'../Results/Outage_Events_Summary_All_Maricopa_{threshold}_{start}-{end}.csv')
filtered_data1=outage_event_df_1[outage_event_df_1[target_variable]>percentile_975]
filtered_data1=filtered_data1.sort_values(by=['year', 'month', 'day'], ascending=[True, True, True])
compare_counties(filtered_data,filtered_data1,target_variable)
#outage_duration_by_month(outage_event_df)
#outage_mag_by_month(outage_event_df)
#plot_seasonal_outages(outage_event_df)
#plot_seasonal_weather_behavior(outage_event_df)
# compare_counties(outage_event_df)
#plot_wind(outage_event_df)
#plot_wind(outage_event_df)
#plot_precip(outage_weather_df)
#plot_max_temp(outage_weather_df)

breakpoint()