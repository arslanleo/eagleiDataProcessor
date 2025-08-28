import pandas as pd

# Load your merged CSV
df = pd.read_csv('merged_massachusetts.csv')

# County centroid coordinates (from US Census Gazetteer)
county_coords = {
    'Barnstable':  (41.706123, -70.164823),
    'Berkshire':   (42.371493, -73.217928),
    'Bristol':     (41.748588, -71.088894),
    'Dukes':       (41.380970, -70.701499),
    'Essex':       (42.642708, -70.864909),
    'Franklin':    (42.584504, -72.591792),
    'Hampden':     (42.136198, -72.635648),
    'Hampshire':   (42.339459, -72.663694),
    'Middlesex':   (42.481718, -71.394916),
    'Nantucket':   (41.293392, -70.102164),
    'Norfolk':     (42.171738, -71.181110),
    'Plymouth':    (41.987196, -70.741942),
    'Suffolk':     (42.338551, -71.018253),
    'Worcester':   (42.311693, -71.940282)
}

# Map coordinates to new columns
df['latitude'] = df['county'].map(lambda c: county_coords.get(c, (None, None))[0])
df['longitude'] = df['county'].map(lambda c: county_coords.get(c, (None, None))[1])

# Save updated file
df.to_csv('merged_massachusetts_with_coords.csv', index=False)

print("Coordinates added and saved to 'merged_massachusetts_with_coords.csv'")
print(df.head())
