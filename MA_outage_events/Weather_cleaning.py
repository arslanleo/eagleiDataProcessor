import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('asos (2).csv')
print(df.head())
print(df.describe())
print(df.columns)
df["sknt"] = pd.to_numeric(df["sknt"].replace("M", 0), errors="coerce")
df["gust"] = pd.to_numeric(df["gust"].replace("M", 0), errors="coerce")
df['tmpf'] = pd.to_numeric(df['tmpf'].replace("M",0), errors = "coerce")

# Print maximum values
print("Max sknt:", df["sknt"].max())
print("Max gust:", df["gust"].max())
print("Max tempf:", df['tmpf'].max())
df['sknt'] = df["sknt"][(df["sknt"] >= 0) & (df["sknt"] <= 150)]
df['gust'] = df["gust"][(df["gust"] >= 0) & (df["gust"] <= 150)]

plt.figure(figsize=(12, 5))

# Histogram for sknt
plt.subplot(1, 2, 1)
plt.hist(df["sknt"].dropna(), bins=50, color="royalblue", edgecolor="black", alpha=0.7)
plt.xlabel("Wind Speed (knots)")
plt.ylabel("Frequency")
plt.title("Histogram of Wind Speed (sknt)")

# Histogram for gust
plt.subplot(1, 2, 2)
plt.hist(df["gust"].dropna(), bins=50, color="darkorange", edgecolor="black", alpha=0.7)
plt.xlabel("Wind Gust (knots)")
plt.ylabel("Frequency")
plt.title("Histogram of Wind Gusts")

plt.tight_layout()
plt.show()



count_sknt_outliers = (df["sknt"] > 150).sum()
count_gust_outliers = (df["gust"] > 150).sum()

print("Number of wind speed entries > 200 knots:", count_sknt_outliers)
print("Number of wind gust entries > 200 knots:", count_gust_outliers)

df.to_csv("asos_clean.csv")
print("Cleaned file saved as asos_clean.csv")