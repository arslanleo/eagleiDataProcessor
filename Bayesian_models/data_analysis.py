import matplotlib.pyplot as plt
import pandas as pd

# inputs - change as needed
state='Rhode Island'
county='Bristol'
start='2018'
end='2019'
threshold=0.001

def plot_wind(df):
    plt.hist(df["gust"].dropna(), bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Wind Gustg + Speed (knots)")
    plt.ylabel("Frequency")
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

# read in data
outage_event_df=pd.read_csv(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.csv')
outage_weather_df=pd.read_csv(f'../Results/Data_All_{county}_{threshold}_{start}-{end}.csv')

#plot_wind(outage_event_df)
#plot_wind(outage_event_df)
#plot_precip(outage_weather_df)
#plot_max_temp(outage_weather_df)

breakpoint()